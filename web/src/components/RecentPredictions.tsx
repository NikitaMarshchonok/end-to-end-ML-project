import { PredictionResponse } from '@/services/api';
import { History, TrendingUp, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface RecentPrediction {
  id: string;
  timestamp: Date;
  modelName: string;
  prediction: PredictionResponse;
}

interface RecentPredictionsProps {
  predictions: RecentPrediction[];
  onClear: () => void;
}

const formatCurrency = (value: number, currency: string) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: currency,
    maximumFractionDigits: 0,
  }).format(value);
};

const formatTime = (date: Date) => {
  return new Intl.DateTimeFormat('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  }).format(date);
};

const RecentPredictions = ({ predictions, onClear }: RecentPredictionsProps) => {
  if (predictions.length === 0) {
    return null;
  }

  return (
    <div className="bg-card rounded-2xl shadow-card p-6 border border-border animate-fade-in">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-lg bg-muted">
            <History className="h-5 w-5 text-muted-foreground" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">Recent Predictions</h3>
            <p className="text-sm text-muted-foreground">Latest predictions</p>
          </div>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={onClear}
          className="text-muted-foreground hover:text-destructive"
        >
          <Trash2 className="h-4 w-4 mr-1" />
          Clear
        </Button>
      </div>

      <div className="space-y-3">
        {predictions.map((item, index) => (
          <div 
            key={item.id}
            className="flex items-center justify-between p-4 rounded-xl bg-muted/50 hover:bg-muted transition-colors animate-slide-up"
            style={{ animationDelay: `${index * 50}ms` }}
          >
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <TrendingUp className="h-4 w-4 text-primary" />
              </div>
              <div>
                <p className="font-medium text-foreground">
                  {formatCurrency(item.prediction.price, item.prediction.currency)}
                </p>
                <p className="text-xs text-muted-foreground">
                  {item.modelName} • {formatTime(item.timestamp)}
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm text-muted-foreground">Range</p>
              <p className="text-xs text-muted-foreground">
                {formatCurrency(item.prediction.p10, item.prediction.currency)} - {formatCurrency(item.prediction.p90, item.prediction.currency)}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default RecentPredictions;
