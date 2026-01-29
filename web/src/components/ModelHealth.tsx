import { MonitoringResponse } from '@/services/api';
import { Activity, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { cn } from '@/lib/utils';

interface ModelHealthProps {
  data: MonitoringResponse | null;
}

const driftLabel = (score: number) => {
  if (score >= 1.0) return { text: 'High', color: 'text-destructive' };
  if (score >= 0.5) return { text: 'Medium', color: 'text-yellow-600' };
  return { text: 'Low', color: 'text-success' };
};

const ModelHealth = ({ data }: ModelHealthProps) => {
  if (!data) return null;

  if (data.note) {
    return (
      <div className="bg-card rounded-2xl shadow-card p-6 border border-border animate-fade-in">
        <div className="flex items-center gap-2 mb-3">
          <div className="p-2 rounded-lg bg-muted">
            <Activity className="h-5 w-5 text-muted-foreground" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">Model Health</h3>
            <p className="text-sm text-muted-foreground">Drift monitoring</p>
          </div>
        </div>
        <div className="text-sm text-muted-foreground">{data.note}</div>
      </div>
    );
  }

  return (
    <div className="bg-card rounded-2xl shadow-card p-6 border border-border animate-fade-in">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-lg bg-muted">
            <Activity className="h-5 w-5 text-muted-foreground" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">Model Health</h3>
            <p className="text-sm text-muted-foreground">Feature drift vs baseline</p>
          </div>
        </div>
        <div className="text-xs text-muted-foreground">
          Recent samples: {data.sample_size}
        </div>
      </div>

      {data.drift.length === 0 ? (
        <div className="text-sm text-muted-foreground">Not enough recent data to compute drift.</div>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Feature</TableHead>
              <TableHead>Baseline mean</TableHead>
              <TableHead>Recent mean</TableHead>
              <TableHead>Drift score</TableHead>
              <TableHead>Level</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.drift.map((item) => {
              const level = driftLabel(item.drift_score);
              return (
                <TableRow key={item.feature}>
                  <TableCell className="font-medium">{item.feature}</TableCell>
                  <TableCell>{item.baseline_mean.toFixed(2)}</TableCell>
                  <TableCell>{item.recent_mean.toFixed(2)}</TableCell>
                  <TableCell>{item.drift_score.toFixed(2)}</TableCell>
                  <TableCell className={cn('font-medium', level.color)}>
                    {level.text}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      )}

      <div className="mt-4 flex items-center gap-2 text-xs text-muted-foreground">
        {data.drift.some((d) => d.drift_score >= 1.0) ? (
          <AlertTriangle className="h-4 w-4 text-destructive" />
        ) : (
          <CheckCircle2 className="h-4 w-4 text-success" />
        )}
        <span>Drift score = |recent_mean − baseline_mean| / baseline_std</span>
      </div>
    </div>
  );
};

export default ModelHealth;
