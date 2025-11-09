
"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { createChart, IChartApi, ISeriesApi, CandlestickData, HistogramData } from 'lightweight-charts';
import { tradingApi, Prediction, Equity, HealthStatus } from "./tradingApi";

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
const PriceChart = ({ data }: { data: HistoricalDataPoint[] }) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!chartContainerRef.current || data.length === 0) return;

        const chart = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
            height: 400,
            layout: { background: { color: '#1a1e25' }, textColor: '#d1d4dc' },
            grid: { vertLines: { color: '#2e333d' }, horzLines: { color: '#2e333d' } },
            timeScale: { timeVisible: true, secondsVisible: false },
        });

        const candlestickSeries = chart.addCandlestickSeries({
            upColor: '#26a69a', downColor: '#ef5350', borderDownColor: '#ef5350',
            borderUpColor: '#26a69a', wickDownColor: '#ef5350', wickUpColor: '#26a69a',
        });

        const volumeSeries = chart.addHistogramSeries({
            color: '#26a69a', priceFormat: { type: 'volume' }, priceScaleId: '',
        });
        chart.priceScale('').applyOptions({ scaleMargins: { top: 0.7, bottom: 0 } });

        const candleData: CandlestickData[] = data.map(d => ({
            time: d.time, open: d.open, high: d.high, low: d.low, close: d.close
        }));
        const volumeData: HistogramData[] = data.map(d => ({
            time: d.time, value: d.volume,
            color: d.close >= d.open ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)',
        }));

        candlestickSeries.setData(candleData);
        volumeSeries.setData(volumeData);
        chart.timeScale().fitContent();

        const handleResize = () => chart.applyOptions({ width: chartContainerRef.current?.clientWidth });
        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, [data]);

    return <div ref={chartContainerRef} className="w-full h-[400px]" />;
};


// ============== Main Dashboard Page ==============
export default function TradingDashboard() {
  const [equities, setEquities] = useState<Equity[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("predictions");
  
  // Modal state
  const [modalData, setModalData] = useState<{prediction: Prediction | null, history: HistoricalDataPoint[]}>({ prediction: null, history: [] });
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Fetch initial data
  const loadData = useCallback(async (refresh = false) => {
    setLoading(true);
    try {
      const [equitiesData, healthData] = await Promise.all([
        tradingApi.getEquities(refresh),
        tradingApi.getHealth(),
      ]);
      setEquities(equitiesData);
      setHealth(healthData);
    } catch (error) {
      console.error("Failed to load initial data:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    const interval = setInterval(() => tradingApi.getHealth().then(setHealth), 10000); // Refresh health every 10s
    return () => clearInterval(interval);
  }, [loadData]);

  // Handle running predictions
  const handlePredict = async (limit: number) => {
    setLoading(true);
    try {
      const preds = await tradingApi.predictAll(limit);
      setPredictions(preds);
      setActiveTab('predictions');
    } catch (error) {
      console.error(`Failed to run predictions for top ${limit}:`, error);
    } finally {
      setLoading(false);
    }
  };

  // Handle opening the details modal
  const handleOpenModal = async (prediction: Prediction) => {
    setIsModalOpen(true);
    setModalData({ prediction, history: [] }); // Open modal immediately
    try {
        const historyData = await tradingApi.getHistory(prediction.ticker);
        setModalData({ prediction, history: historyData });
    } catch (error) {
        console.error("Failed to fetch history:", error);
        // Keep modal open but show error or empty state for chart
    }
  };

  const bullishCount = predictions.filter(p => p.scoring.signal === 'bullish').length;
  const bearishCount = predictions.filter(p => p.scoring.signal === 'bearish').length;

  const renderContent = () => {
    const items = activeTab === 'bullish' ? predictions.filter(p => p.scoring.signal === 'bullish')
                : activeTab === 'bearish' ? predictions.filter(p => p.scoring.signal === 'bearish')
                : activeTab === 'predictions' ? predictions
                : equities;

    if (loading && items.length === 0) return <div className="text-center p-8">Loading...</div>;

    if (items.length === 0) return <div className="text-center p-8">No data available. Run a prediction to begin.</div>;

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {items.map((item: Equity | Prediction, index) => (
            'scoring' in item ? ( // Type guard to check if it's a Prediction
                <div key={index} onClick={() => handleOpenModal(item)} className="p-4 bg-gray-800 rounded-lg cursor-pointer hover:bg-gray-700">
                    <div className="flex justify-between items-center">
                        <h3 className="text-xl font-bold">{item.ticker}</h3>
                        <p className={`font-bold ${item.scoring.signal === 'bullish' ? 'text-green-400' : 'text-red-400'}`}>{item.scoring.signal.toUpperCase()}</p>
                    </div>
                    <p>Price: ${item.price.toFixed(2)}</p>
                    <p>Confidence: {(item.scoring.confidence * 100).toFixed(1)}%</p>
                </div>
            ) : ( // It's an Equity
                 <div key={index} className="p-4 bg-gray-800 rounded-lg">
                    <h3 className="text-xl font-bold">{item.ticker}</h3>
                    <p className="text-sm text-gray-400 truncate">{item.name}</p>
                    <p>Market Cap: ${(item.market_cap / 1e9).toFixed(2)}B</p>
                </div>
            )
        ))}
      </div>
    );
  };
  
  return (
    <div className="bg-gray-900 text-white min-h-screen p-8">
      <header className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Trading Analysis System</h1>
        <p className="text-gray-400">AI-powered equity analysis with real-time predictions</p>
        {health && (
            <div className="flex items-center gap-4 mt-4 text-sm">
                <span>CPU: {health.cpu.toFixed(1)}%</span>
                <span>Memory: {health.memory.toFixed(1)}%</span>
                <span>Equities: {health.equities_count}</span>
                 <span className="text-green-400">Status: {health.status}</span>
            </div>
        )}
      </header>
      
      <div className="flex gap-4 mb-8">
          <button onClick={() => handlePredict(50)} disabled={loading} className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-500 disabled:bg-gray-500">
              {loading ? 'Analyzing...' : 'Predict Top 50'}
          </button>
          <button onClick={() => loadData(true)} disabled={loading} className="px-4 py-2 bg-gray-600 rounded hover:bg-gray-500 disabled:bg-gray-500">
              Refresh Equities
          </button>
      </div>

      <div className="mb-4 border-b border-gray-700">
        <nav className="flex gap-4">
            <button onClick={() => setActiveTab('predictions')} className={`py-2 px-4 ${activeTab === 'predictions' ? 'border-b-2 border-blue-500' : ''}`}>Predictions ({predictions.length})</button>
            <button onClick={() => setActiveTab('bullish')} className={`py-2 px-4 ${activeTab === 'bullish' ? 'border-b-2 border-green-500' : ''}`}>Bullish ({bullishCount})</button>
            <button onClick={() => setActiveTab('bearish')} className={`py-2 px-4 ${activeTab === 'bearish' ? 'border-b-2 border-red-500' : ''}`}>Bearish ({bearishCount})</button>
            <button onClick={() => setActiveTab('equities')} className={`py-2 px-4 ${activeTab === 'equities' ? 'border-b-2 border-gray-500' : ''}`}>All Equities ({equities.length})</button>
        </nav>
      </div>

      {renderContent()}

      {isModalOpen && modalData.prediction && (
          <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4">
              <div className="bg-gray-800 rounded-lg p-6 max-w-4xl w-full">
                  <div className="flex justify-between items-center mb-4">
                      <h2 className="text-2xl font-bold">{modalData.prediction.ticker}</h2>
                      <button onClick={() => setIsModalOpen(false)} className="text-2xl">&times;</button>
                  </div>
                  {modalData.history.length > 0 ? (
                      <PriceChart data={modalData.history} />
                  ) : (
                      <div className="w-full h-[400px] flex items-center justify-center">Loading chart...</div>
                  )}
              </div>
          </div>
      )}
    </div>
  );
}
