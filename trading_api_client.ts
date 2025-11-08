// src/services/tradingApi.ts
// API Client for Trading System Backend

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

export interface Metrics {
  count: number;
  p50: number;
  p95: number;
  p99: number;
  cache_hits: number;
  cache_misses: number;
}

class TradingApiClient {
  private baseURL: string;

  constructor(baseURL: string = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000") {
    this.baseURL = baseURL;
  }

  /**
   * Fetch all available equities
   */
  async getEquities(params?: { refresh?: boolean; limit?: number }): Promise<{
    success: boolean;
    count: number;
    equities: Equity[];
  }> {
    const queryParams = new URLSearchParams();
    if (params?.refresh) queryParams.append("refresh", "true");
    if (params?.limit) queryParams.append("limit", params.limit.toString());

    const response = await fetch(`${this.baseURL}/api/v3/equities?${queryParams}`);
    return response.json();
  }

  /**
   * Force refresh equities from sources
   */
  async refreshEquities(): Promise<{
    success: boolean;
    count: number;
    message: string;
  }> {
    const response = await fetch(`${this.baseURL}/api/v3/equities/refresh`, {
      method: "POST",
    });
    return response.json();
  }

  /**
   * Analyze a single ticker
   */
  async analyze(ticker: string): Promise<Prediction> {
    const response = await fetch(`${this.baseURL}/api/v3/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker }),
    });
    return response.json();
  }

  /**
   * Analyze multiple tickers in batch
   */
  async analyzeBatch(tickers: string[]): Promise<Record<string, Prediction>> {
    const response = await fetch(`${this.baseURL}/api/v3/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tickers }),
    });
    return response.json();
  }

  /**
   * Predict all equities
   */
  async predictAll(limit?: number): Promise<{
    success: boolean;
    count: number;
    predictions: Prediction[];
  }> {
    const response = await fetch(`${this.baseURL}/api/v3/predict/all`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ limit }),
    });
    return response.json();
  }

  /**
   * Stream predictions for all equities (Server-Sent Events)
   */
  async streamPredictions(
    limit: number = 100,
    onProgress: (data: { type: string; index?: number; data?: Prediction; total?: number }) => void
  ): Promise<void> {
    const response = await fetch(`${this.baseURL}/api/v3/predict/stream?limit=${limit}`);
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) throw new Error("No reader available");

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split("\n");

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const jsonStr = line.substring(6);
          try {
            const data = JSON.parse(jsonStr);
            onProgress(data);
          } catch (e) {
            console.error("Failed to parse SSE data:", e);
          }
        }
      }
    }
  }

  /**
   * Stream analysis for a single ticker
   */
  async streamAnalysis(
    ticker: string,
    onProgress: (event: string, data: any) => void
  ): Promise<void> {
    const response = await fetch(`${this.baseURL}/api/v3/stream?ticker=${ticker}`);
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) throw new Error("No reader available");

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split("\n");

      let currentEvent = "";
      let currentData = "";

      for (const line of lines) {
        if (line.startsWith("event: ")) {
          currentEvent = line.substring(7);
        } else if (line.startsWith("data: ")) {
          currentData = line.substring(6);
          if (currentEvent && currentData) {
            try {
              const data = JSON.parse(currentData);
              onProgress(currentEvent, data);
            } catch (e) {
              console.error("Failed to parse SSE data:", e);
            }
            currentEvent = "";
            currentData = "";
          }
        }
      }
    }
  }

  /**
   * Search equities by query
   */
  async searchEquities(query: string): Promise<{
    success: boolean;
    count: number;
    results: Equity[];
  }> {
    const response = await fetch(
      `${this.baseURL}/api/v3/search?q=${encodeURIComponent(query)}`
    );
    return response.json();
  }

  /**
   * Get system health status
   */
  async getHealth(): Promise<HealthStatus> {
    const response = await fetch(`${this.baseURL}/api/v3/health`);
    return response.json();
  }

  /**
   * Get performance metrics
   */
  async getMetrics(): Promise<Metrics> {
    const response = await fetch(`${this.baseURL}/api/v3/metrics`);
    return response.json();
  }
}

// Export singleton instance
export const tradingApi = new TradingApiClient();

// Export class for custom instances
export default TradingApiClient;
