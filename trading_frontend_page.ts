"use client";

import { useState, useEffect } from "react";
import {
  Heading,
  Text,
  Button,
  Column,
  Row,
  Badge,
  Input,
  Grid,
  Card,
  Spinner,
  Icon,
} from "@once-ui-system/core";

interface Equity {
  ticker: string;
  name: string;
  exchange: string;
  sector: string;
  industry: string;
  market_cap: number;
}

interface Prediction {
  ticker: string;
  price: number;
  indicators: {
    rsi: number;
    macd: number;
    sma20: number;
    bollinger: {
      upper: number;
      lower: number;
    };
    volume_spike: boolean;
  };
  scoring: {
    score: number;
    signal: string;
    confidence: number;
  };
  performance: {
    latency_ms: number;
  };
  success: boolean;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

export default function TradingDashboard() {
  const [equities, setEquities] = useState<Equity[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedEquity, setSelectedEquity] = useState<Equity | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [streamingPredictions, setStreamingPredictions] = useState(false);

  // Fetch equities on mount
  useEffect(() => {
    fetchEquities();
  }, []);

  const fetchEquities = async (refresh = false) => {
    setLoading(true);
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v3/equities?limit=100${refresh ? "&refresh=true" : ""}`
      );
      const data = await response.json();
      if (data.success) {
        setEquities(data.equities);
      }
    } catch (error) {
      console.error("Error fetching equities:", error);
    } finally {
      setLoading(false);
    }
  };

  const analyzeSingle = async (ticker: string) => {
    setAnalyzing(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/v3/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker }),
      });
      const data = await response.json();
      if (data.success) {
        setPredictions([data]);
      }
    } catch (error) {
      console.error("Error analyzing:", error);
    } finally {
      setAnalyzing(false);
    }
  };

  const predictAll = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/v3/predict/all`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ limit: 50 }),
      });
      const data = await response.json();
      if (data.success) {
        setPredictions(data.predictions);
      }
    } catch (error) {
      console.error("Error predicting:", error);
    } finally {
      setLoading(false);
    }
  };

  const streamPredictions = async () => {
    setStreamingPredictions(true);
    setPredictions([]);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/v3/predict/stream?limit=50`);
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) return;

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
              if (data.type === "result") {
                setPredictions((prev) => [...prev, data.data]);
              }
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }
    } catch (error) {
      console.error("Error streaming predictions:", error);
    } finally {
      setStreamingPredictions(false);
    }
  };

  const searchEquities = async (query: string) => {
    if (!query) {
      fetchEquities();
      return;
    }
    try {
      const response = await fetch(`${API_BASE_URL}/api/v3/search?q=${encodeURIComponent(query)}`);
      const data = await response.json();
      if (data.success) {
        setEquities(data.results);
      }
    } catch (error) {
      console.error("Error searching:", error);
    }
  };

  const getSignalColor = (signal: string) => {
    return signal === "bullish" ? "green" : "red";
  };

  const filteredEquities = searchQuery
    ? equities.filter(
        (eq) =>
          eq.ticker.toLowerCase().includes(searchQuery.toLowerCase()) ||
          eq.name.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : equities;

  return (
    <Column fillWidth padding="l" gap="l">
      {/* Header */}
      <Column gap="m">
        <Heading variant="display-strong-l">Trading Analysis System</Heading>
        <Text variant="body-default-m" onBackground="neutral-weak">
          AI-powered equity analysis and prediction platform
        </Text>
      </Column>

      {/* Controls */}
      <Row gap="m" fillWidth wrap>
        <Input
          id="search"
          label="Search Equities"
          placeholder="Search by ticker or name..."
          value={searchQuery}
          onChange={(e) => {
            setSearchQuery(e.target.value);
            searchEquities(e.target.value);
          }}
          style={{ flex: 1, minWidth: "300px" }}
        />
        <Button
          onClick={() => fetchEquities(true)}
          disabled={loading}
          variant="secondary"
          prefixIcon="refresh"
        >
          Refresh Equities
        </Button>
        <Button
          onClick={predictAll}
          disabled={loading || streamingPredictions}
          variant="primary"
          prefixIcon="chart"
        >
          {loading ? "Analyzing..." : "Predict All"}
        </Button>
        <Button
          onClick={streamPredictions}
          disabled={loading || streamingPredictions}
          variant="tertiary"
          prefixIcon="stream"
        >
          {streamingPredictions ? "Streaming..." : "Stream Predictions"}
        </Button>
      </Row>

      {/* Stats */}
      <Row gap="m" fillWidth wrap>
        <Card padding="m" style={{ flex: 1, minWidth: "200px" }}>
          <Column gap="s">
            <Text variant="label-default-s" onBackground="neutral-weak">
              Total Equities
            </Text>
            <Heading variant="display-strong-m">{equities.length}</Heading>
          </Column>
        </Card>
        <Card padding="m" style={{ flex: 1, minWidth: "200px" }}>
          <Column gap="s">
            <Text variant="label-default-s" onBackground="neutral-weak">
              Predictions Made
            </Text>
            <Heading variant="display-strong-m">{predictions.length}</Heading>
          </Column>
        </Card>
        <Card padding="m" style={{ flex: 1, minWidth: "200px" }}>
          <Column gap="s">
            <Text variant="label-default-s" onBackground="neutral-weak">
              Bullish Signals
            </Text>
            <Heading variant="display-strong-m" style={{ color: "var(--green-600)" }}>
              {predictions.filter((p) => p.scoring.signal === "bullish").length}
            </Heading>
          </Column>
        </Card>
        <Card padding="m" style={{ flex: 1, minWidth: "200px" }}>
          <Column gap="s">
            <Text variant="label-default-s" onBackground="neutral-weak">
              Bearish Signals
            </Text>
            <Heading variant="display-strong-m" style={{ color: "var(--red-600)" }}>
              {predictions.filter((p) => p.scoring.signal === "bearish").length}
            </Heading>
          </Column>
        </Card>
      </Row>

      {/* Predictions */}
      {predictions.length > 0 && (
        <Column gap="m">
          <Heading variant="heading-strong-l">Latest Predictions</Heading>
          <Grid columns="repeat(auto-fill, minmax(350px, 1fr))" gap="m">
            {predictions.map((pred, idx) => (
              <Card key={`${pred.ticker}-${idx}`} padding="m" border="neutral-medium">
                <Column gap="m">
                  <Row fillWidth horizontal="space-between" vertical="center">
                    <Column gap="xs">
                      <Heading variant="heading-strong-m">{pred.ticker}</Heading>
                      <Text variant="body-default-s" onBackground="neutral-weak">
                        ${pred.price.toFixed(2)}
                      </Text>
                    </Column>
                    <Badge
                      onBackground={
                        pred.scoring.signal === "bullish" ? "green-medium" : "red-medium"
                      }
                    >
                      <Text variant="label-default-s">
                        {pred.scoring.signal.toUpperCase()}
                      </Text>
                    </Badge>
                  </Row>

                  <Column gap="s">
                    <Row fillWidth horizontal="space-between">
                      <Text variant="body-default-s">RSI</Text>
                      <Text variant="body-default-s">{pred.indicators.rsi.toFixed(2)}</Text>
                    </Row>
                    <Row fillWidth horizontal="space-between">
                      <Text variant="body-default-s">MACD</Text>
                      <Text variant="body-default-s">{pred.indicators.macd.toFixed(2)}</Text>
                    </Row>
                    <Row fillWidth horizontal="space-between">
                      <Text variant="body-default-s">SMA20</Text>
                      <Text variant="body-default-s">{pred.indicators.sma20.toFixed(2)}</Text>
                    </Row>
                    <Row fillWidth horizontal="space-between">
                      <Text variant="body-default-s">Confidence</Text>
                      <Text variant="body-default-s">
                        {(pred.scoring.confidence * 100).toFixed(0)}%
                      </Text>
                    </Row>
                    <Row fillWidth horizontal="space-between">
                      <Text variant="body-default-s">Latency</Text>
                      <Text variant="body-default-s">{pred.performance.latency_ms}ms</Text>
                    </Row>
                  </Column>
                </Column>
              </Card>
            ))}
          </Grid>
        </Column>
      )}

      {/* Equities List */}
      <Column gap="m">
        <Heading variant="heading-strong-l">Available Equities</Heading>
        {loading && !streamingPredictions ? (
          <Column fillWidth horizontal="center" padding="xl">
            <Spinner size="l" />
          </Column>
        ) : (
          <Grid columns="repeat(auto-fill, minmax(300px, 1fr))" gap="m">
            {filteredEquities.slice(0, 50).map((equity) => (
              <Card
                key={equity.ticker}
                padding="m"
                border="neutral-medium"
                style={{ cursor: "pointer" }}
                onClick={() => setSelectedEquity(equity)}
              >
                <Column gap="s">
                  <Row fillWidth horizontal="space-between" vertical="center">
                    <Heading variant="heading-strong-s">{equity.ticker}</Heading>
                    <Button
                      size="s"
                      onClick={(e) => {
                        e.stopPropagation();
                        analyzeSingle(equity.ticker);
                      }}
                      disabled={analyzing}
                      variant="tertiary"
                    >
                      {analyzing ? "..." : "Analyze"}
                    </Button>
                  </Row>
                  <Text variant="body-default-s" onBackground="neutral-weak">
                    {equity.name}
                  </Text>
                  {equity.sector && (
                    <Badge onBackground="neutral-medium">
                      <Text variant="label-default-xs">{equity.sector}</Text>
                    </Badge>
                  )}
                  {equity.exchange && (
                    <Text variant="body-default-xs" onBackground="neutral-weak">
                      {equity.exchange}
                    </Text>
                  )}
                </Column>
              </Card>
            ))}
          </Grid>
        )}
      </Column>

      {/* Loading/Streaming Indicator */}
      {(loading || streamingPredictions) && (
        <Row fillWidth horizontal="center" padding="m">
          <Row gap="s" vertical="center">
            <Spinner size="m" />
            <Text variant="body-default-m">
              {streamingPredictions ? "Streaming predictions..." : "Loading..."}
            </Text>
          </Row>
        </Row>
      )}
    </Column>
  );
}
