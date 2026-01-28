import { useState, useEffect } from 'react';
import { Model, Feature } from '@/services/api';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Loader2, Sparkles } from 'lucide-react';

interface PredictionFormProps {
  model: Model;
  onSubmit: (features: Record<string, string | number>) => void;
  isLoading?: boolean;
}

const PredictionForm = ({ model, onSubmit, isLoading }: PredictionFormProps) => {
  const [formValues, setFormValues] = useState<Record<string, string | number>>({});

  useEffect(() => {
    // Reset form when model changes
    setFormValues({});
  }, [model.id]);

  const handleChange = (name: string, value: string | number) => {
    setFormValues(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formValues);
  };

  const renderField = (feature: Feature) => {
    const value = formValues[feature.name] ?? '';

    if (feature.type === 'select' && feature.options) {
      return (
        <Select
          value={String(value)}
          onValueChange={(val) => handleChange(feature.name, val)}
        >
          <SelectTrigger className="h-11 bg-background">
            <SelectValue placeholder={`Select ${feature.label.toLowerCase()}...`} />
          </SelectTrigger>
          <SelectContent>
            {feature.options.map((option) => (
              <SelectItem key={option} value={option}>
                {option}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      );
    }

    return (
      <div className="relative">
        <Input
          type="number"
          value={value}
          onChange={(e) => handleChange(feature.name, e.target.value ? Number(e.target.value) : '')}
          placeholder={feature.placeholder}
          min={feature.min}
          max={feature.max}
          step={feature.step}
          className="h-11 bg-background pr-16"
        />
        {feature.unit && (
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-sm text-muted-foreground">
            {feature.unit}
          </span>
        )}
      </div>
    );
  };

  const isFormValid = model.features.every(f => {
    const val = formValues[f.name];
    return val !== undefined && val !== '';
  });

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {model.features.map((feature) => (
          <div key={feature.name} className="space-y-2">
            <Label htmlFor={feature.name} className="text-sm font-medium">
              {feature.label}
            </Label>
            {renderField(feature)}
          </div>
        ))}
      </div>

      <Button
        type="submit"
        size="lg"
        disabled={!isFormValid || isLoading}
        className="w-full h-12 text-base font-medium gradient-primary hover:opacity-90 transition-opacity"
      >
        {isLoading ? (
          <>
            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
            Analyzing...
          </>
        ) : (
          <>
            <Sparkles className="mr-2 h-5 w-5" />
            Predict Price
          </>
        )}
      </Button>
    </form>
  );
};

export default PredictionForm;
