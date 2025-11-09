
// src/app/(main)/page.tsx - Complete Trading Dashboard
"use client";

import { useState, useEffect, useRef, useCallback } from "react";
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
  Modal, // Assuming Modal component exists in the library
  Card,      // Assuming Card component exists
} from "@once-ui-system/core";
import { useEquities } from "@/hooks/useEquities";
import { usePredictions } from "@/hooks/usePredictions";
import { useDebounce } from "@/hooks/useDebounce";
import { PredictionCard } from "@/components/trading/PredictionCard";
import { EquityCard } from "@/components/trading/EquityCard";
import { StatsCard } from "@/components/trading/StatsCard";
import { SystemHealth } from "@/components/trading/SystemHealth";
import { tradingApi, Prediction, Equity } from "@/services/tradingApi"; // Assuming types are exported from the client
import { createChart, IChartApi, ISeriesApi, CandlestickData, HistogramData } from 'lightweight-charts';

// Define the structure of the historical data points
export interface HistoricalDataPoint {
    time: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

// ============== Price Chart Component ==============
// A dedicated component to render the financial chart
const PriceChart = ({ data, theme = 'dark' }: { data: HistoricalDataPoint[], theme?: 'light' | 'dark' }) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candlestickSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
    const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);

    useEffect(() => {
        if (!chartContainerRef.current || data.length === 0) return;

        // Chart layout options based on theme
        const layoutOptions = {
            background: { color: theme === 'dark' ? '#1a1e25' : '#ffffff' },
            textColor: theme === 'dark' ? '#d1d4dc' : '#333333',
        };

        const gridOptions = {
            vertLines: { color: theme === 'dark' ? '#2e333d' : '#e6e6e6' },
            horzLines: { color: theme === 'dark' ? '#2e333d' : '#e6e6e6' },
        };

        // Create the chart instance
        chartRef.current = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
            height: 400,
            layout: layoutOptions,
            grid: gridOptions,
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
            },
        });

        // Add Candlestick series
        candlestickSeriesRef.current = chartRef.current.addCandlestickSeries({
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderDownColor: '#ef5350',
            borderUpColor: '#26a69a',
            wickDownColor: '#ef5350',
            wickUpColor: '#26a69a',
        });

        // Add Volume series
        volumeSeriesRef.current = chartRef.current.addHistogramSeries({
            color: '#26a69a',
            priceFormat: {
                type: 'volume',
            },
            priceScaleId: '',
        });
        chartRef.current.priceScale('').applyOptions({
           scaleMargins: { top: 0.7, bottom: 0 },
        });

        // Resize observer
        const handleResize = () => {
            if (chartRef.current && chartContainerRef.current) {
                chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };

        window.addEventListener('resize', handleResize);

        // Cleanup
        return () => {
            window.removeEventListener('resize', handleResize);
            if (chartRef.current) {
                chartRef.current.remove();
            }
        };
    }, [theme]);

    // Update chart with new data
    useEffect(() => {
        if (data.length === 0) return;

        const candleData: CandlestickData[] = data.map(d => ({
            time: d.time,
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close
        }));

        const volumeData: HistogramData[] = data.map(d => ({
            time: d.time,
            value: d.volume,
            color: d.close >= d.open ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)',
        }));
        
        if (candlestickSeriesRef.current) {
             candlestickSeriesRef.current.setData(candleData);
        }
       if (volumeSeriesRef.current) {
            volumeSeriesRef.current.setData(volumeData);
       }
       if (chartRef.current) {
           chartRef.current.timeScale().fitContent();
       }

    }, [data]);

    return <div ref={chartContainerRef} style={{ width: '100%', height: '400px' }} />;
};


// ============== Main Dashboard Page ==============
export default function TradingDashboard() {
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState("predictions");
  const debouncedSearch = useDebounce(searchQuery, 300);

  // Modal and Chart State
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedPrediction, setSelectedPrediction] = useState<Prediction | null>(null);
  const [history, setHistory] = useState<HistoricalDataPoint[]>([]);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);

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
  
    // New handler to open the modal and fetch history
    const handlePredictionSelect = useCallback(async (prediction: Prediction) => {
        setSelectedPrediction(prediction);
        setIsModalOpen(true);
        setIsHistoryLoading(true);
        try {
            // This is a conceptual function. You would need to add `getHistory` to your `tradingApi` client.
            // For now, we mock the fetch call.
            const response = await fetch(`http://localhost:5000/api/v3/history?ticker=${prediction.ticker}`);
            if (!response.ok) {
                throw new Error('Failed to fetch historical data');
            }
            const data: HistoricalDataPoint[] = await response.json();
            setHistory(data);
        } catch (error) {
            console.error(error);
            setHistory([]); // Clear history on error
        } finally {
            setIsHistoryLoading(false);
        }
    }, []);

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
              AI-powered equity analysis with real-time predictions and price charts.
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
          {/* ... other buttons ... */}
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
            { id: "predictions", label: `Predictions (${predictions.length})` },
            { id: "equities", label: `Equities (${equities.length})` },
            { id: "bullish", label: `Bullish (${bullishCount})` },
            { id: "bearish", label: `Bearish (${bearishCount})` },
          ]}
        />
        
        {/* Tab Content */}
        <Grid columns="repeat(auto-fill, minmax(350px, 1fr))" gap="m">
            {activeTab === 'predictions' && predictions.map((pred, idx) => (
                <div key={`${pred.ticker}-${idx}`} onClick={() => handlePredictionSelect(pred)}>
                    <PredictionCard prediction={pred} />
                </div>
            ))}
            {activeTab === 'bullish' && predictions.filter(p => p.scoring.signal === 'bullish').map((pred, idx) => (
                 <div key={`${pred.ticker}-${idx}`} onClick={() => handlePredictionSelect(pred)}>
                    <PredictionCard prediction={pred} />
                </div>
            ))}
            {activeTab === 'bearish' && predictions.filter(p => p.scoring.signal === 'bearish').map((pred, idx) => (
                 <div key={`${pred.ticker}-${idx}`} onClick={() => handlePredictionSelect(pred)}>
                    <PredictionCard prediction={pred} />
                </div>
            ))}
            {activeTab === 'equities' && equities.slice(0, 100).map((equity) => (
                <EquityCard
                  key={equity.ticker}
                  equity={equity}
                  onAnalyze={() => analyzeSingle(equity.ticker)}
                  analyzing={predictionsLoading}
                />
             ))}
        </Grid>
      </Column>
      
      {/* Prediction Detail Modal with Chart */}
        {selectedPrediction && (
            <Modal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                title={`${selectedPrediction.ticker} - ${selectedPrediction.price.toFixed(2)}`}
                size="xl" // Assuming a large modal size
            >
                <Column gap="l">
                     <Row horizontal="space-between">
                        <Text variant="heading-strong-l">{selectedPrediction.ticker}</Text>
                        <Text variant="heading-default-l" color={selectedPrediction.scoring.signal === 'bullish' ? 'green' : 'red'}>
                            {selectedPrediction.scoring.signal.toUpperCase()} ({ (selectedPrediction.scoring.confidence * 100).toFixed(1) }%)
                        </Text>
                    </Row>

                    {isHistoryLoading ? (
                        <Row horizontal="center" style={{height: '400px'}} vertical="center">
                            <Spinner size="xl" />
                            <Text>Loading Price History...</Text>
                        </Row>
                    ) : (
                        <PriceChart data={history} />
                    )}
                    
                    <Card padding="l">
                        <Heading variant="heading-default-m" bottomMargin="m">Indicators</Heading>
                        <Grid columns="2" gap="m">
                           <StatsCard label="RSI" value={selectedPrediction.indicators.rsi.toFixed(2)} />
                           <StatsCard label="MACD" value={selectedPrediction.indicators.macd.toFixed(4)} />
                           <StatsCard label="SMA (20)" value={selectedPrediction.indicators.sma20.toFixed(2)} />
                           <StatsCard label="Volume Spike" value={selectedPrediction.indicators.volume_spike ? "Yes" : "No"} />
                        </Grid>
                    </Card>
                </Column>
            </Modal>
        )}

      {/* Footer */}
      <Column fillWidth horizontal="center" padding="xl" gap="s">
        {/* ... footer content ... */}
      </Column>
    </Column>
  );
}
