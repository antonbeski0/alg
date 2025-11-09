
// src/services/tradingApi.ts
import { HistoricalDataPoint } from "@/app/page";

// All data structures are based on the backend (algo.py)
export interface Equity {
  ticker: string;
  name: string;
  exchange: string;
  sector: string;
  industry: string;
  market_cap: number;
}

export interface Indicator {
  rsi: number;
  macd: number;
  sma20: number;
  bollinger: {
    upper: number;
    lower: number;
  };
  volume_spike: boolean;
}

export interface Scoring {
  score: number;
  signal: "bullish" | "bearish";
  confidence: number;
}

export interface Performance {
  latency_ms: number;
}

export interface Prediction {
  ticker: string;
  price: number;
  indicators: Indicator;
  scoring: Scoring;
  performance: Performance;
  success: boolean;
  error?: string;
}

export interface HealthStatus {
  status: string;
  cpu: number;
  memory: number;
  threads: number;
  timestamp: string;
  equities_count: number;
}

// Main API client class
class TradingApiClient {
  private baseURL: string;

  constructor(baseURL?: string) {
    this.baseURL = baseURL || process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";
  }

  // Generic response handler for error checking
  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API Error: ${response.status} - ${errorText}`);
    }
    return response.json();
  }

  // Fetch historical data for charts
  async getHistory(ticker: string): Promise<HistoricalDataPoint[]> {
    const response = await fetch(`${this.baseURL}/api/v3/history?ticker=${ticker}`);
    return this.handleResponse(response);
  }
  
  // Analyze a single ticker
  async analyze(ticker: string): Promise<Prediction> {
    const response = await fetch(`${this.baseURL}/api/v3/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker }),
    });
    return this.handleResponse(response);
  }
  
  // Fetch all equities
  async getEquities(refresh: boolean = false): Promise<Equity[]> {
    const url = `${this.baseURL}/api/v3/equities?refresh=${refresh}`;
    const res = await fetch(url);
    const data = await this.handleResponse<{ equities: Equity[] }>(res);
    return data.equities;
  }

  // Get system health
  async getHealth(): Promise<HealthStatus> {
    const response = await fetch(`${this.baseURL}/api/v3/health`);
    return this.handleResponse(response);
  }

  // Predict all (batch)
  async predictAll(limit: number = 50): Promise<Prediction[]> {
      const response = await fetch(`${this.baseURL}/api/v3/predict/all`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ limit }),
      });
      const data = await this.handleResponse<{ predictions: Prediction[] }>(response);
      return data.predictions;
  }
}

// Export a singleton instance
export const tradingApi = new TradingApiClient();
