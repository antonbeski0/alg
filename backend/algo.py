
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
import logging
import os
import sys
import time
import json
import warnings
import threading
import psutil
from dataclasses import dataclass
from collections import deque
from numba import jit
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import sqlite3
from typing import List, Dict, Optional

warnings.filterwarnings("ignore")

# ==================== CONFIG ====================
@dataclass
class Config:
    MAX_WORKERS: int = 30
    PERIOD: str = "1d"
    INTERVAL: str = "1m"
    LOG_LEVEL: str = "INFO"
    DB_PATH: str = "/tmp/equities_cache.db" # Use /tmp for compatibility with read-only filesystems

config = Config()

# ==================== LOGGER ====================
class PerfLogger:
    def __init__(self):
        self.logger = logging.getLogger("TradingSystem")
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL))

perf = PerfLogger()

# ==================== NUMBA ACCELERATED INDICATORS ====================
@jit(nopython=True, fastmath=True, cache=True)
def rsi(arr, period=14):
    if len(arr) < period + 1: return 50.0
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

# ... (other indicator functions remain the same) ...

# ==================== DATABASE ====================
class EquitiesDB:
    def __init__(self, db_path=config.DB_PATH):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_db()

    def init_db(self):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS equities (
                    ticker TEXT PRIMARY KEY, name TEXT, exchange TEXT,
                    sector TEXT, industry TEXT, market_cap REAL
                )
            """)
            conn.commit()
    
    def save_equities(self, equities: List[Dict]):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO equities (ticker, name, exchange, sector, industry, market_cap)
                VALUES (:ticker, :name, :exchange, :sector, :industry, :market_cap)
            """, equities)
            conn.commit()

    def get_all_equities(self) -> List[Dict]:
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM equities ORDER BY market_cap DESC")
            return [dict(row) for row in cursor.fetchall()]

equities_db = EquitiesDB()

# ==================== DATA FETCHER ====================
class EquitiesFetcher:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

    def _fetch_from_wikipedia(self, url: str, table_index: int, symbol_col: str) -> List[str]:
        try:
            tables = pd.read_html(url)
            df = tables[table_index]
            return df[symbol_col].str.replace('.', '-').tolist()
        except Exception as e:
            perf.logger.error(f"Failed to fetch from {url}: {e}")
            return []

    def get_ticker_info(self, ticker: str) -> Optional[Dict]:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                "ticker": ticker,
                "name": info.get("longName", ticker),
                "exchange": info.get("exchange", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", 0)
            }
        except Exception:
            return None # yfinance often fails on some tickers, so we silently ignore

    def fetch_all_equities(self, refresh=False) -> List[Dict]:
        if not refresh and equities_db.get_all_equities():
            perf.logger.info("Loaded equities from cache.")
            return equities_db.get_all_equities()

        perf.logger.info("Fetching fresh equities list...")
        sp500 = self._fetch_from_wikipedia("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", 0, 'Symbol')
        nasdaq100 = self._fetch_from_wikipedia("https://en.wikipedia.org/wiki/Nasdaq-100", 4, 'Ticker')
        all_tickers = sorted(list(set(sp500 + nasdaq100)))

        equities = []
        with self.executor as ex:
            futures = {ex.submit(self.get_ticker_info, t): t for t in all_tickers}
            for future in as_completed(futures):
                if result := future.result():
                    equities.append(result)
        
        if equities:
            equities_db.save_equities(equities)
            perf.logger.info(f"Saved {len(equities)} equities to database.")
        return equities

fetcher = EquitiesFetcher()

# ==================== ANALYZER ====================
class Analyzer:
    def analyze(self, ticker: str) -> Dict:
        try:
            df = yf.download(ticker, period=config.PERIOD, interval=config.INTERVAL, progress=False)
            if df.empty: return {"success": False, "error": "No data from yfinance"}
            
            close = df["Close"].values.astype(np.float64)
            val_rsi = rsi(close)
            score = 0
            if val_rsi < 30: score += 1
            if val_rsi > 70: score -= 1
            signal = "bullish" if score > 0 else "bearish"

            return {
                "ticker": ticker,
                "price": float(close[-1]),
                "indicators": {"rsi": float(val_rsi)},
                "scoring": {"signal": signal, "confidence": abs(score)},
                "success": True
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict_batch(self, tickers: List[str]) -> List[Dict]:
        results = []
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures = [executor.submit(self.analyze, ticker) for ticker in tickers]
            for future in as_completed(futures):
                result = future.result()
                if result and result.get("success"):
                    results.append(result)
        return results

analyzer = Analyzer()

# ==================== FLASK API ====================
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Restrict to /api path

@app.route("/api/v3/health")
def api_health():
    return jsonify({
        "status": "ok",
        "cpu": psutil.cpu_percent(),
        "memory": psutil.virtual_memory().percent,
        "equities_count": len(equities_db.get_all_equities())
    })

@app.route("/api/v3/equities")
def api_get_equities():
    refresh = request.args.get("refresh", "false").lower() == "true"
    equities = fetcher.fetch_all_equities(refresh=refresh)
    return jsonify({"success": True, "equities": equities})

@app.route("/api/v3/predict/all", methods=["POST"])
def api_predict_all():
    limit = request.json.get("limit", 50)
    equities = equities_db.get_all_equities()
    tickers = [eq["ticker"] for eq in equities[:limit]]
    predictions = analyzer.predict_batch(tickers)
    return jsonify({"success": True, "predictions": predictions})

@app.route("/api/v3/history")
def api_history():
    ticker = request.args.get("ticker")
    if not ticker: return jsonify({"success": False, "error": "Ticker required"}), 400
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y", interval="1d")
        if hist.empty: return jsonify({"success": False, "error": "No history data"}), 404
        
        hist_data = [{
            "time": index.strftime('%Y-%m-%d'), "open": row["Open"], "high": row["High"],
            "low": row["Low"], "close": row["Close"], "volume": row["Volume"]
        } for index, row in hist.iterrows()]
        
        return jsonify(hist_data)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ==================== STARTUP & RUN ====================
with app.app_context():
    # Load equities on startup in a background thread
    threading.Thread(target=fetcher.fetch_all_equities, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
