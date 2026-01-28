import { Building2, TrendingUp, Shield } from 'lucide-react';

const HeroSection = () => {
  return (
    <section className="relative overflow-hidden gradient-surface pb-8 pt-16 md:pb-12 md:pt-24">
      {/* Decorative background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 rounded-full bg-primary/5 blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 rounded-full bg-primary/5 blur-3xl" />
      </div>

      <div className="container relative">
        <div className="mx-auto max-w-3xl text-center">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-1.5 text-sm font-medium text-primary mb-6 animate-fade-in">
            <TrendingUp className="h-4 w-4" />
            <span>AI-Powered Valuations</span>
          </div>

          <h1 className="text-4xl font-bold tracking-tight text-foreground sm:text-5xl md:text-6xl animate-slide-up">
            Real Estate Price
            <span className="block text-primary">Prediction</span>
          </h1>

          <p className="mt-6 text-lg text-muted-foreground max-w-2xl mx-auto animate-slide-up">
            Get accurate property valuations powered by machine learning. 
            Enter your property details and receive instant predictions with confidence intervals.
          </p>

          <div className="mt-10 flex flex-wrap items-center justify-center gap-6 text-sm text-muted-foreground animate-fade-in">
            <div className="flex items-center gap-2">
              <Building2 className="h-5 w-5 text-primary" />
              <span>Multiple Markets</span>
            </div>
            <div className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              <span>Confidence Ranges</span>
            </div>
            <div className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-primary" />
              <span>Explainable AI</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
