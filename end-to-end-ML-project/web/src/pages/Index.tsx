import { useState, useEffect } from 'react';
import HeroSection from '@/components/HeroSection';
import ModelSelector from '@/components/ModelSelector';
import PredictionForm from '@/components/PredictionForm';
import PriceResultCard from '@/components/PriceResultCard';
import ExplainabilitySection from '@/components/ExplainabilitySection';
import RecentPredictions from '@/components/RecentPredictions';
import LoadingState from '@/components/LoadingState';
import ErrorState from '@/components/ErrorState';
import {
  Model,
  PredictionResponse,
  fetchModels,
  predictPrice,
  mockModels,
  generateMockPrediction,
} from '@/services/api';

interface RecentPrediction {
  id: string;
  timestamp: Date;
  modelName: string;
  prediction: PredictionResponse;
}

const USE_MOCK_DATA = true; // Toggle for development

const Index = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [isLoadingModels, setIsLoadingModels] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);

  const [currentPrediction, setCurrentPrediction] = useState<PredictionResponse | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  const [recentPredictions, setRecentPredictions] = useState<RecentPrediction[]>([]);

  const selectedModel = models.find(m => m.id === selectedModelId);

  useEffect(() => {
    loadModels();
  }, []);

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

  const handlePredict = async (features: Record<string, string | number>) => {
    if (!selectedModelId || !selectedModel) return;

    setIsPredicting(true);
    setPredictionError(null);

    try {
      let result: PredictionResponse;

      if (USE_MOCK_DATA) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1200));
        result = generateMockPrediction(selectedModelId, features);
      } else {
        result = await predictPrice({
          model_id: selectedModelId,
          features,
        });
      }

      setCurrentPrediction(result);

      // Add to recent predictions
      const newPrediction: RecentPrediction = {
        id: crypto.randomUUID(),
        timestamp: new Date(),
        modelName: selectedModel.name,
        prediction: result,
      };

      setRecentPredictions(prev => [newPrediction, ...prev].slice(0, 10));
    } catch (error) {
      setPredictionError('Failed to get prediction. Please try again.');
    } finally {
      setIsPredicting(false);
    }
  };

  const handleClearHistory = () => {
    setRecentPredictions([]);
  };

  const handleModelSelect = (modelId: string) => {
    setSelectedModelId(modelId);
    setCurrentPrediction(null);
    setPredictionError(null);
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
            <div className="mt-8 grid gap-6 md:grid-cols-2">
              <PriceResultCard prediction={currentPrediction} />
              <ExplainabilitySection factors={currentPrediction.factors} />
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
