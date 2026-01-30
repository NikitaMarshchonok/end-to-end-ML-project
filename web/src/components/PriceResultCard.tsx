import { useState } from 'react';
import { PredictionResponse } from '@/services/api';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { TrendingUp, Info } from 'lucide-react';

interface PriceResultCardProps {
  prediction: PredictionResponse;
  onSubmitFeedback?: (actualPrice: number) => Promise<{ abs_error: number; pct_error: number | null }>;
}

const formatCurrency = (value: number, currency: string) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: currency,
    maximumFractionDigits: 0,
  }).format(value);
};

const PriceResultCard = ({ prediction, onSubmitFeedback }: PriceResultCardProps) => {
  const [actualPrice, setActualPrice] = useState('');
  const [feedbackStatus, setFeedbackStatus] = useState<'idle' | 'submitting' | 'success' | 'error'>('idle');
  const [feedbackMessage, setFeedbackMessage] = useState<string | null>(null);

  const handleSubmitFeedback = async () => {
    if (!onSubmitFeedback) return;
    const value = Number(actualPrice);
    if (!value || value <= 0) {
      setFeedbackStatus('error');
      setFeedbackMessage('Please enter a valid actual price.');
      return;
    }
    setFeedbackStatus('submitting');
    setFeedbackMessage(null);
    try {
      const result = await onSubmitFeedback(value);
      const pct = result.pct_error !== null ? `${(result.pct_error * 100).toFixed(1)}%` : '—';
      setFeedbackStatus('success');
      setFeedbackMessage(`Saved. Abs error: ${formatCurrency(result.abs_error, prediction.currency)} • MAPE: ${pct}`);
    } catch (e) {
      setFeedbackStatus('error');
      setFeedbackMessage('Failed to save feedback. Please try again.');
    }
  };

  return (
    <div className="animate-scale-in">
      <div className="bg-card rounded-2xl shadow-card p-6 md:p-8 border border-border">
        {/* Header badges */}
        <div className="flex flex-wrap items-center gap-2 mb-6">
          <Badge variant="secondary" className="font-medium">
            Model v{prediction.model_version}
          </Badge>
          <Badge variant="outline" className="font-medium">
            {prediction.currency}
          </Badge>
        </div>

        {/* Main price */}
        <div className="text-center mb-8">
          <p className="text-sm font-medium text-muted-foreground mb-2 flex items-center justify-center gap-1">
            <TrendingUp className="h-4 w-4" />
            Predicted Price
          </p>
          <p className="text-4xl md:text-5xl font-bold text-foreground tracking-tight">
            {formatCurrency(prediction.price, prediction.currency)}
          </p>
        </div>

        {/* Confidence range */}
        <div className="gradient-surface rounded-xl p-5">
          <div className="flex items-center gap-1 mb-4">
            <Info className="h-4 w-4 text-muted-foreground" />
            <p className="text-sm font-medium text-muted-foreground">
              Confidence Range
            </p>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <p className="text-xs text-muted-foreground mb-1">P10 (Low)</p>
              <p className="text-lg font-semibold text-foreground">
                {formatCurrency(prediction.p10, prediction.currency)}
              </p>
            </div>
            <div className="text-center border-x border-border">
              <p className="text-xs text-muted-foreground mb-1">P50 (Median)</p>
              <p className="text-lg font-semibold text-primary">
                {formatCurrency(prediction.p50, prediction.currency)}
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-muted-foreground mb-1">P90 (High)</p>
              <p className="text-lg font-semibold text-foreground">
                {formatCurrency(prediction.p90, prediction.currency)}
              </p>
            </div>
          </div>

          {/* Visual range bar */}
          <div className="mt-4 relative h-2 bg-border rounded-full overflow-hidden">
            <div 
              className="absolute h-full gradient-primary rounded-full transition-all duration-500"
              style={{ 
                left: '10%', 
                right: '10%',
              }}
            />
            <div 
              className="absolute h-full w-1 bg-foreground rounded-full"
              style={{ left: '50%', transform: 'translateX(-50%)' }}
            />
          </div>
        </div>

        {/* Feedback loop */}
        <div className="mt-6 rounded-xl border border-border bg-muted/30 p-4">
          <p className="text-sm font-medium text-foreground mb-2">Feedback (actual sale price)</p>
          {prediction.prediction_id ? (
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <Input
                type="number"
                value={actualPrice}
                onChange={(e) => setActualPrice(e.target.value)}
                placeholder="Enter actual price"
                className="sm:max-w-[220px]"
              />
              <Button
                type="button"
                onClick={handleSubmitFeedback}
                disabled={!onSubmitFeedback || feedbackStatus === 'submitting'}
                className="sm:w-auto"
              >
                {feedbackStatus === 'submitting' ? 'Saving...' : 'Save feedback'}
              </Button>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">
              Feedback is available when prediction history is enabled.
            </p>
          )}
          {feedbackMessage && (
            <p
              className={
                feedbackStatus === 'success'
                  ? 'mt-2 text-xs text-success'
                  : 'mt-2 text-xs text-destructive'
              }
            >
              {feedbackMessage}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default PriceResultCard;
