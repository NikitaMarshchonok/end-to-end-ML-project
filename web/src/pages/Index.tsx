import { useState, useEffect } from 'react';
import HeroSection from '@/components/HeroSection';
import ModelSelector from '@/components/ModelSelector';
import PredictionForm from '@/components/PredictionForm';
import PriceResultCard from '@/components/PriceResultCard';
import ExplainabilitySection from '@/components/ExplainabilitySection';
import RecentPredictions from '@/components/RecentPredictions';
import ComparableSales from '@/components/ComparableSales';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import LoadingState from '@/components/LoadingState';
import ErrorState from '@/components/ErrorState';
import {
  Model,
  PredictionResponse,
  ExplainResponse,
  ComparablesResponse,
  fetchModels,
  predictPrice,
  explainPrediction,
  fetchPredictions,
  clearPredictions,
  fetchComparables,
  mockModels,
  generateMockPrediction,
  generateMockExplain,
  generateMockComparables,
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
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
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

  const [recentPredictions, setRecentPredictions] = useState<RecentPrediction[]>([]);

  const selectedModel = models.find(m => m.id === selectedModelId);

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    if (models.length > 0) {
      loadHistory();
    }
  }, [models]);

  const loadModels = async () => {
    setIsLoadingModels(true);
    setModelsError(null);

    try {
      if (USE_MOCK_DATA) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 500));
        setModels(mockModels);
        if (mockModels.length > 0) {
          setSelectedModelId(mockModels[0].id);
        }
      } else {
        const data = await fetchModels();
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
      const mapped: RecentPrediction[] = items.map((item) => {
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
    if (!selectedModelId || !selectedModel) return;

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
        result = generateMockPrediction(selectedModelId, features);
        explain = generateMockExplain(selectedModelId, features);
        comparables = generateMockComparables(selectedModelId, features, topK);
      } else {
        result = await predictPrice({
          model_id: selectedModelId,
          features,
        });
        explain = await explainPrediction({
          model_id: selectedModelId,
          features,
        });
        comparables = await fetchComparables(
          {
            model_id: selectedModelId,
            features,
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

  useEffect(() => {
    if (!lastFeatures || !lastModelId || !currentPrediction) return;

    const topK = Number(comparablesCount) || 5;
    if (USE_MOCK_DATA) {
      setCurrentComparables(generateMockComparables(lastModelId, lastFeatures, topK));
      return;
    }

    fetchComparables(
      {
        model_id: lastModelId,
        features: lastFeatures,
      },
      topK
    )
      .then(setCurrentComparables)
      .catch(() => {});
  }, [comparablesCount, lastFeatures, lastModelId, currentPrediction]);

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
                <PriceResultCard prediction={currentPrediction} />
                <ExplainabilitySection
                  factors={currentExplain?.factors || currentPrediction.factors}
                  ranges={currentExplain?.ranges}
                  currency={currentExplain?.currency || currentPrediction.currency}
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
