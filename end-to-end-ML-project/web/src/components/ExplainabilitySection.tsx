import { Factor } from '@/services/api';
import { ArrowUp, ArrowDown, Lightbulb } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ExplainabilitySectionProps {
  factors: Factor[];
}

const ExplainabilitySection = ({ factors }: ExplainabilitySectionProps) => {
  const sortedFactors = [...factors].sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));

  return (
    <div className="bg-card rounded-2xl shadow-card p-6 border border-border animate-slide-up">
      <div className="flex items-center gap-2 mb-6">
        <div className="p-2 rounded-lg bg-primary/10">
          <Lightbulb className="h-5 w-5 text-primary" />
        </div>
        <div>
          <h3 className="font-semibold text-foreground">Price Factors</h3>
          <p className="text-sm text-muted-foreground">Top factors influencing the prediction</p>
        </div>
      </div>

      <div className="space-y-4">
        {sortedFactors.map((factor, index) => (
          <div 
            key={factor.name} 
            className="flex items-center gap-4 animate-fade-in"
            style={{ animationDelay: `${index * 100}ms` }}
          >
            {/* Direction indicator */}
            <div className={cn(
              "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center",
              factor.direction === 'up' 
                ? "bg-success/10 text-success" 
                : "bg-destructive/10 text-destructive"
            )}>
              {factor.direction === 'up' ? (
                <ArrowUp className="h-4 w-4" />
              ) : (
                <ArrowDown className="h-4 w-4" />
              )}
            </div>

            {/* Factor details */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between mb-1">
                <p className="text-sm font-medium text-foreground truncate">
                  {factor.name}
                </p>
                <span className={cn(
                  "text-sm font-semibold",
                  factor.direction === 'up' ? "text-success" : "text-destructive"
                )}>
                  {factor.direction === 'up' ? '+' : ''}{factor.impact}%
                </span>
              </div>

              {/* Impact bar */}
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div 
                  className={cn(
                    "h-full rounded-full transition-all duration-700 ease-out",
                    factor.direction === 'up' 
                      ? "bg-success" 
                      : "bg-destructive"
                  )}
                  style={{ width: `${Math.abs(factor.impact)}%` }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ExplainabilitySection;
