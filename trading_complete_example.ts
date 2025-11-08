// src/app/(main)/page.tsx - Complete Trading Dashboard
"use client";

import { useState } from "react";
import {
  Heading,
  Text,
  Button,
  Column,
  Row,
  Input,
  Grid,
  Tabs,
  Spinner,
} from "@once-ui-system/core";
import { useEquities } from "@/hooks/useEquities";
import { usePredictions } from "@/hooks/usePredictions";
import { useDebounce } from "@/hooks/useDebounce";
import { PredictionCard } from "@/components/trading/PredictionCard";
import { EquityCard } from "@/components/trading/EquityCard";
import { StatsCard } from "@/components/trading/StatsCard";
import { SystemHealth } from "@/components/trading/SystemHealth";

export default function TradingDashboard() {
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState("predictions");
  const debouncedSearch = useDebounce(searchQuery, 300);

  // Custom hooks
  const {
    equities,
    loading: equitiesLoading,
    refreshEquities,
    searchEquities,
  } = useEquities();

  const {
    predictions,
    loading: predictionsLoading,
    streaming,
    analyzeSingle,
    predictAll,
    streamPredictions,
    clearPredictions,
  } = usePredictions();

  // Search handler
  const handleSearch = (query: string) => {
    setSearchQuery(query);
    if (query.length > 2) {
      searchEquities(query);
    }
  };

  // Stats calculations
  const bullishCount = predictions.filter((p) => p.scoring.signal === "bullish").length;
  const bearishCount = predictions.filter((p) => p.scoring.signal === "bearish").length;
  const avgConfidence =
    predictions.length > 0
      ? predictions.reduce((acc, p) => acc + p.scoring.confidence, 0) / predictions.length
      : 0;
  const avgLatency =
    predictions.length > 0
      ? predictions.reduce((acc, p) => acc + p.performance.latency_ms, 0) / predictions.length
      : 0;

  return (
    <Column fillWidth padding="xl" gap="xl" style={{ minHeight: "100vh" }}>
      {/* Header */}
      <Column gap="m" maxWidth="xl" fillWidth>
        <Row fillWidth horizontal="space-between" vertical="center" wrap>
          <Column gap="s">
            <Heading variant="display-strong-xl">Trading Analysis System</Heading>
            <Text variant="body-default-l" onBackground="neutral-weak">
              AI-powered equity analysis with real-time predictions
            </Text>
          </Column>
          <SystemHealth />
        </Row>
      </Column>

      {/* Main Stats */}
      <Row gap="m" fillWidth wrap maxWidth="xl">
        <StatsCard label="Total Equities" value={equities.length} />
        <StatsCard label="Predictions" value={predictions.length} />
        <StatsCard label="Bullish Signals" value={bullishCount} color="var(--green-600)" />
        <StatsCard label="Bearish Signals" value={bearishCount} color="var(--red-600)" />
        <StatsCard
          label="Avg Confidence"
          value={`${(avgConfidence * 100).toFixed(1)}%`}
        />
        <StatsCard label="Avg Latency" value={`${avgLatency.toFixed(1)}ms`} />
      </Row>

      {/* Controls */}
      <Column gap="m" fillWidth maxWidth="xl">
        <Row gap="m" fillWidth wrap>
          <Input
            id="search"
            label="Search"
            placeholder="Search by ticker, name, sector..."
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            style={{ flex: 1, minWidth: "300px" }}
          />
          <Button
            onClick={() => refreshEquities()}
            disabled={equitiesLoading}
            variant="secondary"
          >
            {equitiesLoading ? <Spinner size="s" /> : "Refresh Equities"}
          </Button>
        </Row>

        <Row gap="m" fillWidth wrap>
          <Button
            onClick={() => predictAll(50)}
            disabled={predictionsLoading || streaming}
            variant="primary"
            size="l"
          >
            {predictionsLoading ? <Spinner size="s" /> : "Predict Top 50"}
          </Button>
          <Button
            onClick={() => predictAll(100)}
            disabled={predictionsLoading || streaming}
            variant="primary"
            size="l"
          >
            {predictionsLoading ? <Spinner size="s" /> : "Predict Top 100"}
          </Button>
          <Button
            onClick={() => streamPredictions(50)}
            disabled={predictionsLoading || streaming}
            variant="secondary"
            size="l"
          >
            {streaming ? <Spinner size="s" /> : "Stream 50"}
          </Button>
          <Button
            onClick={clearPredictions}
            disabled={predictions.length === 0}
            variant="tertiary"
            size="l"
          >
            Clear All
          </Button>
        </Row>
      </Column>

      {/* Tabs */}
      <Column fillWidth maxWidth="xl" gap="l">
        <Tabs
          value={activeTab}
          onValueChange={setActiveTab}
          tabs={[
            {
              id: "predictions",
              label: `Predictions (${predictions.length})`,
            },
            {
              id: "equities",
              label: `Equities (${equities.length})`,
            },
            {
              id: "bullish",
              label: `Bullish (${bullishCount})`,
            },
            {
              id: "bearish",
              label: `Bearish (${bearishCount})`,
            },
          ]}
        />

        {/* Predictions Tab */}
        {activeTab === "predictions" && (
          <Column gap="l">
            {streaming && (
              <Row fillWidth horizontal="center" padding="m">
                <Row gap="s" vertical="center">
                  <Spinner size="m" />
                  <Text variant="body-default-m">
                    Streaming predictions... ({predictions.length} received)
                  </Text>
                </Row>
              </Row>
            )}

            {predictions.length === 0 && !streaming && (
              <Column fillWidth horizontal="center" padding="xl">
                <Text variant="body-default-l" onBackground="neutral-weak">
                  No predictions yet. Click "Predict" to start analyzing equities.
                </Text>
              </Column>
            )}

            <Grid columns="repeat(auto-fill, minmax(350px, 1fr))" gap="m">
              {predictions.map((pred, idx) => (
                <PredictionCard
                  key={`${pred.ticker}-${idx}`}
                  prediction={pred}
                  onAnalyze={() => analyzeSingle(pred.ticker)}
                />
              ))}
            </Grid>
          </Column>
        )}

        {/* Equities Tab */}
        {activeTab === "equities" && (
          <Column gap="l">
            {equitiesLoading ? (
              <Column fillWidth horizontal="center" padding="xl">
                <Spinner size="l" />
              </Column>
            ) : (
              <>
                <Text variant="body-default-m" onBackground="neutral-weak">
                  Showing {equities.length} equities
                  {searchQuery && ` matching "${searchQuery}"`}
                </Text>
                <Grid columns="repeat(auto-fill, minmax(300px, 1fr))" gap="m">
                  {equities.slice(0, 100).map((equity) => (
                    <EquityCard
                      key={equity.ticker}
                      equity={equity}
                      onAnalyze={() => analyzeSingle(equity.ticker)}
                      analyzing={predictionsLoading}
                    />
                  ))}
                </Grid>
              </>
            )}
          </Column>
        )}

        {/* Bullish Tab */}
        {activeTab === "bullish" && (
          <Column gap="l">
            {bullishCount === 0 ? (
              <Column fillWidth horizontal="center" padding="xl">
                <Text variant="body-default-l" onBackground="neutral-weak">
                  No bullish signals yet.
                </Text>
              </Column>
            ) : (
              <Grid columns="repeat(auto-fill, minmax(350px, 1fr))" gap="m">
                {predictions
                  .filter((p) => p.scoring.signal === "bullish")
                  .map((pred, idx) => (
                    <PredictionCard
                      key={`${pred.ticker}-${idx}`}
                      prediction={pred}
                      onAnalyze={() => analyzeSingle(pred.ticker)}
                    />
                  ))}
              </Grid>
            )}
          </Column>
        )}

        {/* Bearish Tab */}
        {activeTab === "bearish" && (
          <Column gap="l">
            {bearishCount === 0 ? (
              <Column fillWidth horizontal="center" padding="xl">
                <Text variant="body-default-l" onBackground="neutral-weak">
                  No bearish signals yet.
                </Text>
              </Column>
            ) : (
              <Grid columns="repeat(auto-fill, minmax(350px, 1fr))" gap="m">
                {predictions
                  .filter((p) => p.scoring.signal === "bearish")
                  .map((pred, idx) => (
                    <PredictionCard
                      key={`${pred.ticker}-${idx}`}
                      prediction={pred}
                      onAnalyze={() => analyzeSingle(pred.ticker)}
                    />
                  ))}
              </Grid>
            )}
          </Column>
        )}
      </Column>

      {/* Footer */}
      <Column fillWidth horizontal="center" padding="xl" gap="s">
        <Text variant="body-default-s" onBackground="neutral-weak">
          Powered by Military-Grade Trading System
        </Text>
        <Text variant="body-default-xs" onBackground="neutral-weak">
          Real-time equity analysis with AI-powered predictions
        </Text>
      </Column>
    </Column>
  );
}
