import { Loader2 } from 'lucide-react';

interface LoadingStateProps {
  message?: string;
}

const LoadingState = ({ message = 'Loading...' }: LoadingStateProps) => {
  return (
    <div className="flex flex-col items-center justify-center py-12 animate-fade-in">
      <Loader2 className="h-8 w-8 text-primary animate-spin mb-4" />
      <p className="text-muted-foreground text-sm">{message}</p>
    </div>
  );
};

export default LoadingState;
