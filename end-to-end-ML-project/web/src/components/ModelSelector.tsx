import { Model } from '@/services/api';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Database } from 'lucide-react';

interface ModelSelectorProps {
  models: Model[];
  selectedModelId: string | null;
  onSelect: (modelId: string) => void;
  isLoading?: boolean;
}

const ModelSelector = ({ models, selectedModelId, onSelect, isLoading }: ModelSelectorProps) => {
  const selectedModel = models.find(m => m.id === selectedModelId);

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-foreground flex items-center gap-2">
        <Database className="h-4 w-4 text-primary" />
        Select Model
      </label>
      <Select
        value={selectedModelId || ''}
        onValueChange={onSelect}
        disabled={isLoading}
      >
        <SelectTrigger className="w-full h-12 bg-background border-border hover:border-primary/50 transition-colors">
          <SelectValue placeholder="Choose a prediction model..." />
        </SelectTrigger>
        <SelectContent>
          {models.map((model) => (
            <SelectItem key={model.id} value={model.id} className="py-3">
              <div className="flex flex-col items-start">
                <span className="font-medium">{model.name}</span>
                <span className="text-xs text-muted-foreground">
                  v{model.version} • {model.features.length} features
                </span>
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      
      {selectedModel && (
        <p className="text-xs text-muted-foreground mt-1">
          Currency: {selectedModel.currency}
        </p>
      )}
    </div>
  );
};

export default ModelSelector;
