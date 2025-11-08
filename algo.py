# Military-Grade Trading System - ENHANCED WITH ALL EQUITIES
# ✅ Ultra-low latency + Full Production Features + All Equities Fetching
# Author: BE SKY & GPT-5

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
import requests
import warnings
import threading
import psutil
from dataclasses import dataclass
from collections import deque
from numba import jit
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from datetime import datetime
import sqlite3
from typing import List, Dict, Optional

warnings.filterwarnings("ignore")
load_dotenv()

# ==================== CONFIG ====================
@dataclass
class Config:
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1/chat/completions"
    MAX_WORKERS: int = 30
    REQUEST_TIMEOUT: int = 3
    PERIOD: str = "1d"
    INTERVAL: str = "1m"
    CACHE_TTL: int = 20
    CACHE_SIZE: int = 2000
    FAILURE_THRESHOLD: int = 5
    RECOVERY_TIMEOUT: int = 60
    LOG_LEVEL: str = "INFO"
    DB_PATH: str = "equities_cache.db"

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
        self.metrics = {
            "latencies": deque(maxlen=10000),
            "errors": deque(maxlen=1000),
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self.lock = threading.Lock()

    def log_latency(self, duration):
        with self.lock:
            self.metrics["latencies"].append(duration)

    def log_error(self, err, ctx):
        with self.lock:
            self.metrics["errors"].append({"time": time.time(), "error": err, "context": ctx})
        self.logger.error(f"{err} | Context: {ctx}")

perf = PerfLogger()

# ==================== NUMBA INDICATORS ====================
@jit(nopython=True, fastmath=True, cache=True)
def rsi(arr, period=14):
    if len(arr) < period + 1: return 50.0
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain*(period-1)+gains[i])/period
        avg_loss = (avg_loss*(period-1)+losses[i])/period
    return 100.0 if avg_loss == 0 else 100.0 - (100/(1+avg_gain/avg_loss))

@jit(nopython=True, fastmath=True, cache=True)
def macd(arr):
    if len(arr) < 26: return 0
    ema12 = np.mean(arr[-12:])
    ema26 = np.mean(arr[-26:])
    return ema12 - ema26

@jit(nopython=True, fastmath=True, cache=True)
def sma(arr, period=20):
    if len(arr) < period: return np.mean(arr)
    return np.mean(arr[-period:])

@jit(nopython=True, fastmath=True, cache=True)
def bollinger(arr, period=20, num_std=2):
    if len(arr) < period: return (0,0)
    mean = np.mean(arr[-period:])
    std = np.std(arr[-period:])
    return mean+num_std*std, mean-num_std*std

# ==================== EQUITIES DATABASE ====================
class EquitiesDB:
    def __init__(self, db_path=config.DB_PATH):
        self.db_path = db_path
        self.init_db()
        self.cache_lock = threading.Lock()
        
    def init_db(self):
        """Initialize SQLite database for equities caching"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equities (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                exchange TEXT,
                sector TEXT,
                industry TEXT,
                market_cap REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                ticker TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                price REAL,
                prediction TEXT,
                confidence REAL,
                PRIMARY KEY (ticker, timestamp)
            )
        """)
        conn.commit()
        conn.close()
        
    def save_equities(self, equities: List[Dict]):
        """Save equities to database"""
        with self.cache_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for eq in equities:
                cursor.execute("""
                    INSERT OR REPLACE INTO equities 
                    (ticker, name, exchange, sector, industry, market_cap, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    eq.get('ticker', ''),
                    eq.get('name', ''),
                    eq.get('exchange', ''),
                    eq.get('sector', ''),
                    eq.get('industry', ''),
                    eq.get('market_cap', 0)
                ))
            conn.commit()
            conn.close()
    
    def get_all_equities(self, limit: Optional[int] = None) -> List[Dict]:
        """Retrieve all equities from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query = "SELECT ticker, name, exchange, sector, industry, market_cap FROM equities"
        if limit:
            query += f" LIMIT {limit}"
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "ticker": row[0],
                "name": row[1],
                "exchange": row[2],
                "sector": row[3],
                "industry": row[4],
                "market_cap": row[5]
            }
            for row in rows
        ]
    
    def save_prediction(self, ticker: str, price: float, prediction: str, confidence: float):
        """Save prediction to database"""
        with self.cache_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (ticker, price, prediction, confidence)
                VALUES (?, ?, ?, ?)
            """, (ticker, price, prediction, confidence))
            conn.commit()
            conn.close()

equities_db = EquitiesDB()

# ==================== EQUITIES FETCHER ====================
class EquitiesFetcher:
    """Fetch comprehensive list of equities from various sources"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        
    def fetch_sp500(self) -> List[str]:
        """Fetch S&P 500 tickers"""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            return df['Symbol'].str.replace('.', '-').tolist()
        except Exception as e:
            perf.log_error(str(e), {"source": "sp500"})
            return []
    
    def fetch_nasdaq100(self) -> List[str]:
        """Fetch NASDAQ 100 tickers"""
        try:
            url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            tables = pd.read_html(url)
            df = tables[4]  # The correct table index
            return df['Ticker'].tolist()
        except Exception as e:
            perf.log_error(str(e), {"source": "nasdaq100"})
            return []
    
    def fetch_dow30(self) -> List[str]:
        """Fetch DOW 30 tickers"""
        try:
            url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
            tables = pd.read_html(url)
            df = tables[1]
            return df['Symbol'].tolist()
        except Exception as e:
            perf.log_error(str(e), {"source": "dow30"})
            return []
    
    def fetch_popular_tickers(self) -> List[str]:
        """Additional popular tickers"""
        return [
            # Crypto
            "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
            # Major Forex
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
            # Commodities
            "GC=F", "SI=F", "CL=F", "NG=F",
            # Indices
            "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX"
        ]
    
    def get_ticker_info(self, ticker: str) -> Optional[Dict]:
        """Fetch detailed information for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                "ticker": ticker,
                "name": info.get("longName", ticker),
                "exchange": info.get("exchange", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0)
            }
        except Exception as e:
            perf.log_error(str(e), {"ticker": ticker, "action": "get_info"})
            return None
    
    def fetch_all_equities(self, refresh=False) -> List[Dict]:
        """Fetch all equities from multiple sources"""
        if not refresh:
            # Try to load from database first
            cached = equities_db.get_all_equities()
            if cached:
                perf.logger.info(f"Loaded {len(cached)} equities from cache")
                return cached
        
        perf.logger.info("Fetching fresh equities data...")
        
        # Collect all tickers
        all_tickers = set()
        all_tickers.update(self.fetch_sp500())
        all_tickers.update(self.fetch_nasdaq100())
        all_tickers.update(self.fetch_dow30())
        all_tickers.update(self.fetch_popular_tickers())
        
        perf.logger.info(f"Collected {len(all_tickers)} unique tickers")
        
        # Fetch detailed info concurrently
        equities = []
        with self.executor as ex:
            futures = {ex.submit(self.get_ticker_info, t): t for t in all_tickers}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    equities.append(result)
        
        # Save to database
        if equities:
            equities_db.save_equities(equities)
            perf.logger.info(f"Saved {len(equities)} equities to database")
        
        return equities

fetcher = EquitiesFetcher()

# ==================== ANALYZER ====================
class Analyzer:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

    def fetch(self, ticker, interval=None, period=None):
        try:
            df = yf.download(ticker, period=period or config.PERIOD, interval=interval or config.INTERVAL, progress=False)
            return df if df is not None and not df.empty else None
        except Exception as e:
            perf.log_error(str(e), {"ticker": ticker})
            return None

    def analyze(self, ticker, stream=False):
        start = time.perf_counter_ns()
        df = self.fetch(ticker)
        if df is None:
            return {"success": False, "error": "no_data", "ticker": ticker}

        close = df["Close"].values.astype(np.float64)
        vol = df["Volume"].values.astype(np.float64)

        # Indicators
        val_rsi = rsi(close)
        if stream: yield "progress", {"step": "RSI", "value": float(val_rsi)}

        val_macd = macd(close)
        if stream: yield "progress", {"step": "MACD", "value": float(val_macd)}

        val_sma = sma(close, 20)
        val_boll_up, val_boll_low = bollinger(close)
        if stream: yield "progress", {"step": "Bollinger", "upper": float(val_boll_up), "lower": float(val_boll_low)}

        avg_vol = np.mean(vol[-20:])
        vol_spike = vol[-1] > 2*avg_vol
        if stream: yield "progress", {"step": "Volume", "spike": bool(vol_spike)}

        # Scoring
        score = 0
        if val_rsi < 30: score += 1
        if val_rsi > 70: score -= 1
        if val_macd > 0: score += 1
        if vol_spike: score += 1
        signal = "bullish" if score > 0 else "bearish"
        confidence = abs(score) / 4.0  # Normalize to 0-1
        
        if stream: yield "progress", {"step": "Scoring", "score": score, "signal": signal}

        latency = round((time.perf_counter_ns()-start)/1e6,3)
        perf.log_latency(latency)

        result = {
            "ticker": ticker,
            "price": float(close[-1]),
            "indicators": {
                "rsi": float(val_rsi),
                "macd": float(val_macd),
                "sma20": float(val_sma),
                "bollinger": {"upper": float(val_boll_up), "lower": float(val_boll_low)},
                "volume_spike": bool(vol_spike),
            },
            "scoring": {"score": score, "signal": signal, "confidence": confidence},
            "performance": {"latency_ms": latency},
            "success": True
        }
        
        # Save prediction
        equities_db.save_prediction(ticker, float(close[-1]), signal, confidence)

        if stream: yield "final", result
        else: return result
    
    def predict_batch(self, tickers: List[str], limit: Optional[int] = None) -> List[Dict]:
        """Predict for multiple tickers concurrently"""
        if limit:
            tickers = tickers[:limit]
        
        results = []
        with self.executor as ex:
            futures = {ex.submit(self.analyze, t): t for t in tickers}
            for future in as_completed(futures):
                result = future.result()
                if result and result.get("success"):
                    results.append(result)
        
        return results

analyzer = Analyzer()

# ==================== FLASK API ====================
app = Flask(__name__)
CORS(app)

@app.route("/")
def root():
    return jsonify({
        "status": "ok", 
        "name": "Military-Grade Trading System - Enhanced",
        "version": "3.0",
        "features": ["analysis", "streaming", "batch", "all_equities", "predictions"]
    })

@app.route("/api/v3/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json() or {}
    ticker = data.get("ticker", "").upper()
    if not ticker:
        return jsonify({"success": False, "error": "ticker_required"}), 400
    return jsonify(analyzer.analyze(ticker))

@app.route("/api/v3/stream")
def api_stream():
    ticker = request.args.get("ticker", "").upper()
    if not ticker:
        return jsonify({"success": False, "error": "ticker_required"}), 400

    def gen():
        for ev, payload in analyzer.analyze(ticker, stream=True):
            yield f"event: {ev}\ndata: {json.dumps(payload)}\n\n"
    return Response(stream_with_context(gen()), mimetype="text/event-stream")

@app.route("/api/v3/batch", methods=["POST"])
def api_batch():
    data = request.get_json() or {}
    tickers = data.get("tickers", [])
    results = {}
    with analyzer.executor as ex:
        futures = {ex.submit(analyzer.analyze, t): t for t in tickers}
        for f in as_completed(futures):
            results[futures[f]] = f.result()
    return jsonify(results)

@app.route("/api/v3/equities", methods=["GET"])
def api_get_equities():
    """Get all available equities"""
    refresh = request.args.get("refresh", "false").lower() == "true"
    limit = request.args.get("limit", type=int)
    
    equities = fetcher.fetch_all_equities(refresh=refresh)
    
    if limit:
        equities = equities[:limit]
    
    return jsonify({
        "success": True,
        "count": len(equities),
        "equities": equities
    })

@app.route("/api/v3/equities/refresh", methods=["POST"])
def api_refresh_equities():
    """Force refresh equities from sources"""
    equities = fetcher.fetch_all_equities(refresh=True)
    return jsonify({
        "success": True,
        "count": len(equities),
        "message": f"Refreshed {len(equities)} equities"
    })

@app.route("/api/v3/predict/all", methods=["POST"])
def api_predict_all():
    """Analyze and predict all available equities"""
    data = request.get_json() or {}
    limit = data.get("limit", 100)  # Default to 100 to prevent overload
    
    equities = equities_db.get_all_equities(limit=limit)
    tickers = [eq["ticker"] for eq in equities]
    
    perf.logger.info(f"Starting prediction for {len(tickers)} equities")
    results = analyzer.predict_batch(tickers, limit=limit)
    
    return jsonify({
        "success": True,
        "count": len(results),
        "predictions": results
    })

@app.route("/api/v3/predict/stream")
def api_predict_stream():
    """Stream predictions for all equities"""
    limit = request.args.get("limit", 100, type=int)
    
    def gen():
        equities = equities_db.get_all_equities(limit=limit)
        tickers = [eq["ticker"] for eq in equities]
        
        yield f"data: {json.dumps({'type': 'start', 'total': len(tickers)})}\n\n"
        
        for i, ticker in enumerate(tickers):
            result = analyzer.analyze(ticker)
            if result and result.get("success"):
                yield f"data: {json.dumps({'type': 'result', 'index': i, 'data': result})}\n\n"
        
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"
    
    return Response(stream_with_context(gen()), mimetype="text/event-stream")

@app.route("/api/v3/health")
def api_health():
    return jsonify({
        "status": "ok",
        "cpu": psutil.cpu_percent(),
        "memory": psutil.virtual_memory().percent,
        "threads": threading.active_count(),
        "timestamp": datetime.utcnow().isoformat(),
        "equities_count": len(equities_db.get_all_equities())
    })

@app.route("/api/v3/metrics")
def api_metrics():
    latencies = list(perf.metrics["latencies"])
    def pct(p): return round(np.percentile(latencies, p),3) if latencies else 0
    return jsonify({
        "count": len(latencies),
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "cache_hits": perf.metrics["cache_hits"],
        "cache_misses": perf.metrics["cache_misses"]
    })

@app.route("/api/v3/search", methods=["GET"])
def api_search_equities():
    """Search equities by name, ticker, sector, or industry"""
    query = request.args.get("q", "").lower()
    if not query:
        return jsonify({"success": False, "error": "query_required"}), 400
    
    all_equities = equities_db.get_all_equities()
    results = [
        eq for eq in all_equities
        if query in eq["ticker"].lower() 
        or query in eq["name"].lower()
        or query in eq.get("sector", "").lower()
        or query in eq.get("industry", "").lower()
    ]
    
    return jsonify({
        "success": True,
        "count": len(results),
        "results": results
    })

# ==================== STARTUP ====================
@app.before_request
def startup():
    """Initialize equities on first request"""
    if not hasattr(app, 'initialized'):
        perf.logger.info("Initializing equities database...")
        # Load equities in background
        threading.Thread(target=fetcher.fetch_all_equities, daemon=True).start()
        app.initialized = True

# ==================== RUN ====================
if __name__ == "__main__":
    perf.logger.info("Starting Military-Grade Trading System...")
    perf.logger.info("Initializing equities database...")
    
    # Pre-load equities on startup
    threading.Thread(target=fetcher.fetch_all_equities, daemon=True).start()
    
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
