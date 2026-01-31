import { useState, useEffect } from 'react';
import HeroSection from '@/components/HeroSection';
import ModelSelector from '@/components/ModelSelector';
import PredictionForm from '@/components/PredictionForm';
import PriceResultCard from '@/components/PriceResultCard';
import ExplainabilitySection from '@/components/ExplainabilitySection';
import RecentPredictions from '@/components/RecentPredictions';
import ComparableSales from '@/components/ComparableSales';
import ModelHealth from '@/components/ModelHealth';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import LoadingState from '@/components/LoadingState';
import ErrorState from '@/components/ErrorState';
import {
  Model,
  Market,
  PredictionResponse,
  ExplainResponse,
  ComparablesResponse,
  MonitoringResponse,
  MetricsResponse,
  MetricsTimeseriesItem,
  fetchModels,
  fetchMarkets,
  resolveMarket,
  predictPrice,
  explainPrediction,
  fetchPredictions,
  clearPredictions,
  fetchComparables,
  fetchMonitoring,
  fetchMetrics,
  fetchMetricsTimeseries,
  submitFeedback,
  mockMarkets,
  mockModels,
  generateMockPrediction,
  generateMockExplain,
  generateMockComparables,
  generateMockMonitoring,
  generateMockFeedback,
  generateMockMetrics,
  generateMockMetricsTimeseries,
} from '@/services/api';

interface RecentPrediction {
  id: string;
  timestamp: Date;
  modelName: string;
  prediction: PredictionResponse;
}

const USE_MOCK_DATA = import.meta.env.VITE_USE_MOCK_DATA === 'true';

const Index = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [markets, setMarkets] = useState<Market[]>([]);
  const [selectedMarketId, setSelectedMarketId] = useState<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [areaUnit, setAreaUnit] = useState<'m2' | 'sqft'>('m2');
  const [displayCurrency, setDisplayCurrency] = useState<string | null>(null);
  const [geoLat, setGeoLat] = useState('');
  const [geoLong, setGeoLong] = useState('');
  const [geoStatus, setGeoStatus] = useState<string | null>(null);
  const [isLoadingModels, setIsLoadingModels] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);

  const [currentPrediction, setCurrentPrediction] = useState<PredictionResponse | null>(null);
  const [currentExplain, setCurrentExplain] = useState<ExplainResponse | null>(null);
  const [currentComparables, setCurrentComparables] = useState<ComparablesResponse | null>(null);
  const [comparablesCount, setComparablesCount] = useState('5');
  const [lastFeatures, setLastFeatures] = useState<Record<string, string | number> | null>(null);
  const [lastModelId, setLastModelId] = useState<string | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [modelHealth, setModelHealth] = useState<MonitoringResponse | null>(null);
  const [modelMetrics, setModelMetrics] = useState<MetricsResponse | null>(null);
  const [metricsSeries, setMetricsSeries] = useState<MetricsTimeseriesItem[] | null>(null);

  const [recentPredictions, setRecentPredictions] = useState<RecentPrediction[]>([]);

  const selectedModel = models.find(m => m.id === selectedModelId);
  const baseCurrency = markets.find(m => m.id === selectedMarketId)?.currency;
  const currencyOptions = Array.from(
    new Set(
      [baseCurrency, 'USD', 'EUR', 'ILS', 'TWD'].filter(Boolean) as string[]
    )
  );

  const applyMockDisplayCurrency = (prediction: PredictionResponse) => {
    if (!displayCurrency || displayCurrency === prediction.currency) {
      return prediction;
    }
    const rates: Record<string, Record<string, number>> = {
      ILS: { USD: 0.27, EUR: 0.25, TWD: 8.6 },
      USD: { ILS: 3.7, EUR: 0.92, TWD: 31.8 },
      EUR: { ILS: 4.0, USD: 1.08, TWD: 34.5 },
      TWD: { ILS: 0.12, USD: 0.032, EUR: 0.029 },
    };
    const rate = rates[prediction.currency]?.[displayCurrency] || 1;
    return {
      ...prediction,
      display_currency: displayCurrency,
      display_price: prediction.price * rate,
      display_p10: prediction.p10 * rate,
      display_p50: prediction.p50 * rate,
      display_p90: prediction.p90 * rate,
    };
  };

  useEffect(() => {
    loadMarkets();
  }, []);

  useEffect(() => {
    if (models.length > 0) {
      loadHistory();
    }
  }, [models, selectedMarketId]);

  useEffect(() => {
    if (selectedMarketId) {
      loadModels(selectedMarketId);
    }
  }, [selectedMarketId]);

  useEffect(() => {
    if (!selectedMarketId) return;
    const market = markets.find(m => m.id === selectedMarketId);
    if (market && !displayCurrency) {
      setDisplayCurrency(market.currency);
    }
  }, [selectedMarketId, markets, displayCurrency]);

  const loadMarkets = async () => {
    try {
      if (USE_MOCK_DATA) {
        setMarkets(mockMarkets);
        setSelectedMarketId(mockMarkets[0]?.id || null);
      } else {
        const data = await fetchMarkets();
        setMarkets(data);
        setSelectedMarketId(data[0]?.id || null);
      }
    } catch {
      setMarkets([]);
      setSelectedMarketId(null);
    }
  };

  const loadModels = async (marketId?: string | null) => {
    setIsLoadingModels(true);
    setModelsError(null);

    try {
      if (USE_MOCK_DATA) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 500));
        const filtered = marketId ? mockModels.filter(m => m.market_id === marketId) : mockModels;
        setModels(filtered);
        if (filtered.length > 0) {
          setSelectedModelId(filtered[0].id);
        }
      } else {
        const data = await fetchModels(marketId || undefined);
        setModels(data);
        if (data.length > 0) {
          setSelectedModelId(data[0].id);
        }
      }
    } catch (error) {
      setModelsError('Failed to load prediction models. Please try again.');
    } finally {
      setIsLoadingModels(false);
    }
  };

  const loadHistory = async () => {
    try {
      if (USE_MOCK_DATA) {
        return;
      }
      const items = await fetchPredictions(10);
      const filtered = selectedMarketId
        ? items.filter(item => item.market_id === selectedMarketId)
        : items;
      const mapped: RecentPrediction[] = filtered.map((item) => {
        const modelName = models.find(m => m.id === item.model_id)?.name || item.model_id;
        return {
          id: String(item.id),
          timestamp: new Date(item.created_at),
          modelName,
          prediction: {
            price: item.price,
            p10: item.p10,
            p50: item.p50,
            p90: item.p90,
            model_version: item.model_version,
            currency: item.currency,
            factors: item.factors,
          },
        };
      });
      setRecentPredictions(mapped);
    } catch {
      // Ignore history errors to keep UI responsive
    }
  };

  const handlePredict = async (features: Record<string, string | number>) => {
    if (!selectedModelId || !selectedModel || !selectedMarketId) return;

    setIsPredicting(true);
    setPredictionError(null);
    setCurrentExplain(null);
    setCurrentComparables(null);

    try {
      let result: PredictionResponse;
      let explain: ExplainResponse;
      let comparables: ComparablesResponse;
      const topK = Number(comparablesCount) || 5;

      if (USE_MOCK_DATA) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1200));
        result = applyMockDisplayCurrency(generateMockPrediction(selectedModelId, features));
        explain = generateMockExplain(selectedModelId, features);
        comparables = generateMockComparables(selectedModelId, features, topK);
      } else {
        result = await predictPrice({
          market_id: selectedMarketId,
          model_id: selectedModelId,
          features,
          area_unit: areaUnit,
          display_currency: displayCurrency || undefined,
        });
        explain = await explainPrediction({
          market_id: selectedMarketId,
          model_id: selectedModelId,
          features,
          area_unit: areaUnit,
          display_currency: displayCurrency || undefined,
        });
        comparables = await fetchComparables(
          {
            market_id: selectedMarketId,
            model_id: selectedModelId,
            features,
            area_unit: areaUnit,
            display_currency: displayCurrency || undefined,
          },
          topK
        );
      }

      setCurrentPrediction(result);
      setCurrentExplain(explain);
      setCurrentComparables(comparables);
      setLastFeatures(features);
      setLastModelId(selectedModelId);

      // Add to recent predictions
      const newPrediction: RecentPrediction = {
        id: crypto.randomUUID(),
        timestamp: new Date(),
        modelName: selectedModel.name,
        prediction: result,
      };

      setRecentPredictions(prev => [newPrediction, ...prev].slice(0, 10));
      await loadHistory();
    } catch (error) {
      setPredictionError('Failed to get prediction. Please try again.');
    } finally {
      setIsPredicting(false);
    }
  };

  const handleClearHistory = () => {
    if (USE_MOCK_DATA) {
      setRecentPredictions([]);
      return;
    }
    clearPredictions()
      .then(() => loadHistory())
      .catch(() => setRecentPredictions([]));
  };

  const handleModelSelect = (modelId: string) => {
    setSelectedModelId(modelId);
    setCurrentPrediction(null);
    setCurrentExplain(null);
    setCurrentComparables(null);
    setPredictionError(null);
    setLastFeatures(null);
    setLastModelId(null);
  };

  const handleMarketSelect = (marketId: string) => {
    setSelectedMarketId(marketId);
    setSelectedModelId(null);
    setModels([]);
    setCurrentPrediction(null);
    setCurrentExplain(null);
    setCurrentComparables(null);
    setPredictionError(null);
    setLastFeatures(null);
    setLastModelId(null);
    setRecentPredictions([]);
    setModelHealth(null);
    setModelMetrics(null);
    const market = markets.find(m => m.id === marketId);
    setDisplayCurrency(market?.currency || null);
    setGeoStatus(null);
  };

  const handleResolveMarket = async () => {
    const lat = Number(geoLat);
    const long = Number(geoLong);
    if (!lat || !long) {
      setGeoStatus('Enter valid coordinates.');
      return;
    }
    try {
      const result = await resolveMarket(lat, long);
      setSelectedMarketId(result.market_id);
      setGeoStatus(`Detected: ${result.market_name} (${result.distance_km.toFixed(0)} km)`);
    } catch {
      setGeoStatus('Failed to detect market. Please try again.');
    }
  };

  useEffect(() => {
    if (!lastFeatures || !lastModelId || !currentPrediction || !selectedMarketId) return;

    const topK = Number(comparablesCount) || 5;
    if (USE_MOCK_DATA) {
      setCurrentComparables(generateMockComparables(lastModelId, lastFeatures, topK));
      return;
    }

    fetchComparables(
      {
        market_id: selectedMarketId,
        model_id: lastModelId,
        features: lastFeatures,
        area_unit: areaUnit,
        display_currency: displayCurrency || undefined,
      },
      topK
    )
      .then(setCurrentComparables)
      .catch(() => {});
  }, [comparablesCount, lastFeatures, lastModelId, currentPrediction, selectedMarketId, areaUnit, displayCurrency]);

  useEffect(() => {
    if (!selectedModelId || !selectedMarketId) return;
    if (USE_MOCK_DATA) {
      setModelHealth(generateMockMonitoring(selectedModelId));
      setModelMetrics(generateMockMetrics());
      setMetricsSeries(generateMockMetricsTimeseries());
      return;
    }
    fetchMonitoring(selectedModelId, selectedMarketId)
      .then(setModelHealth)
      .catch(() => setModelHealth(null));
    fetchMetrics(selectedModelId, selectedMarketId)
      .then(setModelMetrics)
      .catch(() => setModelMetrics(null));
    fetchMetricsTimeseries(selectedModelId, selectedMarketId)
      .then(setMetricsSeries)
      .catch(() => setMetricsSeries(null));
  }, [selectedModelId, selectedMarketId]);

  const handleSubmitFeedback = async (actualPrice: number, actualCurrency?: string) => {
    if (!currentPrediction?.prediction_id) {
      throw new Error('Missing prediction id');
    }

    if (USE_MOCK_DATA) {
      const feedback = generateMockFeedback();
      setModelMetrics(generateMockMetrics());
      setMetricsSeries(generateMockMetricsTimeseries());
      return feedback;
    }

    const feedback = await submitFeedback(currentPrediction.prediction_id, actualPrice, actualCurrency);
    fetchMetrics(selectedModelId || undefined, selectedMarketId || undefined)
      .then(setModelMetrics)
      .catch(() => {});
    fetchMetricsTimeseries(selectedModelId || undefined, selectedMarketId || undefined)
      .then(setMetricsSeries)
      .catch(() => {});
    return feedback;
  };

  return (
    <div className="min-h-screen bg-background">
      <HeroSection />

      <main className="container pb-16">
        <div className="max-w-4xl mx-auto">
          {/* Main prediction card */}
          <div className="bg-card rounded-2xl shadow-elevated border border-border p-6 md:p-8 -mt-6 relative z-10">
            {isLoadingModels ? (
              <LoadingState message="Loading prediction models..." />
            ) : modelsError ? (
              <ErrorState message={modelsError} onRetry={loadModels} />
            ) : (
              <div className="space-y-6">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    Market
                  </label>
                  <Select
                    value={selectedMarketId || ''}
                    onValueChange={handleMarketSelect}
                    disabled={markets.length === 0}
                  >
                    <SelectTrigger className="w-full h-12 bg-background border-border hover:border-primary/50 transition-colors">
                      <SelectValue placeholder="Choose a market..." />
                    </SelectTrigger>
                    <SelectContent>
                      {markets.map((market) => (
                        <SelectItem key={market.id} value={market.id} className="py-3">
                          <div className="flex flex-col items-start">
                            <span className="font-medium">{market.name}</span>
                            <span className="text-xs text-muted-foreground">
                              Currency: {market.currency}
                            </span>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-[1fr_1fr_auto]">
                    <Input
                      type="number"
                      placeholder="Latitude"
                      value={geoLat}
                      onChange={(e) => setGeoLat(e.target.value)}
                    />
                    <Input
                      type="number"
                      placeholder="Longitude"
                      value={geoLong}
                      onChange={(e) => setGeoLong(e.target.value)}
                    />
                    <Button type="button" variant="secondary" onClick={handleResolveMarket}>
                      Detect
                    </Button>
                  </div>
                  {geoStatus && (
                    <p className="text-xs text-muted-foreground mt-2">{geoStatus}</p>
                  )}
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    Area unit
                  </label>
                  <Select value={areaUnit} onValueChange={(value) => setAreaUnit(value as 'm2' | 'sqft')}>
                    <SelectTrigger className="w-full h-12 bg-background border-border hover:border-primary/50 transition-colors">
                      <SelectValue placeholder="Choose unit..." />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="m2">m²</SelectItem>
                      <SelectItem value="sqft">sq ft</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    Display currency
                  </label>
                  <Select
                    value={displayCurrency || baseCurrency || ''}
                    onValueChange={setDisplayCurrency}
                  >
                    <SelectTrigger className="w-full h-12 bg-background border-border hover:border-primary/50 transition-colors">
                      <SelectValue placeholder="Choose currency..." />
                    </SelectTrigger>
                    <SelectContent>
                      {currencyOptions.map((cur) => (
                        <SelectItem key={cur} value={cur}>
                          {cur}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <ModelSelector
                  models={models}
                  selectedModelId={selectedModelId}
                  onSelect={handleModelSelect}
                />

                {selectedModel && (
                  <div className="pt-4 border-t border-border">
                    <PredictionForm
                      model={selectedModel}
                      onSubmit={handlePredict}
                      isLoading={isPredicting}
                      areaUnit={areaUnit}
                    />
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Prediction error */}
          {predictionError && (
            <div className="mt-6">
              <ErrorState 
                message={predictionError} 
                onRetry={() => setPredictionError(null)} 
              />
            </div>
          )}

          {/* Results section */}
          {currentPrediction && !predictionError && (
            <>
              <div className="mt-8 grid gap-6 md:grid-cols-2">
                <PriceResultCard
                  prediction={currentPrediction}
                  onSubmitFeedback={handleSubmitFeedback}
                  displayCurrency={displayCurrency}
                  currencyOptions={currencyOptions}
                />
                <ExplainabilitySection
                  factors={currentExplain?.factors || currentPrediction.factors}
                  ranges={currentExplain?.display_ranges || currentExplain?.ranges}
                  currency={currentExplain?.display_currency || currentExplain?.currency || currentPrediction.currency}
                />
              </div>

              <div className="mt-6 flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-foreground">Comparable Sales</p>
                  <p className="text-xs text-muted-foreground">Choose how many similar listings to show</p>
                </div>
                <div className="w-32">
                  <Select value={comparablesCount} onValueChange={setComparablesCount}>
                    <SelectTrigger className="h-9">
                      <SelectValue placeholder="Top K" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="3">Top 3</SelectItem>
                      <SelectItem value="5">Top 5</SelectItem>
                      <SelectItem value="10">Top 10</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </>
          )}

          {currentPrediction && currentComparables && !predictionError && (
            <div className="mt-6">
              <ComparableSales data={currentComparables} />
            </div>
          )}

          {/* Recent predictions */}
          <div className="mt-8">
            <RecentPredictions
              predictions={recentPredictions}
              onClear={handleClearHistory}
            />
          </div>

          <div className="mt-8">
            <ModelHealth data={modelHealth} metrics={modelMetrics} timeseries={metricsSeries} />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border py-6">
        <div className="container">
          <p className="text-center text-sm text-muted-foreground">
            Predictions are estimates and should not be used as the sole basis for financial decisions.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
