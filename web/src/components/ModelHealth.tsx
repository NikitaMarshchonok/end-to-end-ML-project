import { MetricsResponse, MetricsTimeseriesItem, MonitoringResponse } from '@/services/api';
import { Activity, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { cn } from '@/lib/utils';
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from '@/components/ui/chart';
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from 'recharts';

interface ModelHealthProps {
  data: MonitoringResponse | null;
  metrics?: MetricsResponse | null;
  timeseries?: MetricsTimeseriesItem[] | null;
}

const driftLabel = (score: number) => {
  if (score >= 1.0) return { text: 'High', color: 'text-destructive', bg: 'bg-destructive/10' };
  if (score >= 0.5) return { text: 'Medium', color: 'text-yellow-600', bg: 'bg-yellow-500/10' };
  return { text: 'Low', color: 'text-success', bg: 'bg-success/10' };
};

const formatNumber = (value: number | null, digits = 0) => {
  if (value === null || value === undefined) return '—';
  return value.toFixed(digits);
};

const formatPct = (value: number | null) => {
  if (value === null || value === undefined) return '—';
  return `${(value * 100).toFixed(1)}%`;
};

const ModelHealth = ({ data, metrics, timeseries }: ModelHealthProps) => {
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

  const highCount = data.drift.filter((d) => d.drift_score >= 1.0).length;
  const mediumCount = data.drift.filter((d) => d.drift_score >= 0.5 && d.drift_score < 1.0).length;

  const chartData =
    timeseries?.map((item) => ({
      bucket: item.bucket,
      mae: item.mae,
      mape: item.mape !== null ? item.mape * 100 : null,
    })) || [];

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

      <div
        className={cn(
          'mb-4 rounded-xl border px-4 py-3 text-sm',
          highCount > 0 ? 'border-destructive/30 bg-destructive/5 text-destructive' : 'border-border bg-muted/30 text-muted-foreground'
        )}
      >
        {highCount > 0 ? (
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            <span>
              Drift alert: {highCount} feature{highCount > 1 ? 's' : ''} exceed the high threshold (≥ 1.0).
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-success" />
            <span>No high drift detected for recent data.</span>
          </div>
        )}
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
                <TableCell>
                  <span className={cn('inline-flex items-center rounded-full px-2 py-1 text-xs font-semibold', level.color, level.bg)}>
                    {level.text}
                  </span>
                </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      )}

      <div className="mt-4 rounded-xl border border-border bg-muted/20 p-4 text-sm">
        <div className="mb-2 font-medium text-foreground">Feedback metrics (real prices)</div>
        {metrics && metrics.count > 0 ? (
          <div className="grid grid-cols-3 gap-3 text-xs text-muted-foreground">
            <div>
              <div className="text-foreground font-semibold">MAE</div>
              <div>{formatNumber(metrics.mae, 0)}</div>
            </div>
            <div>
              <div className="text-foreground font-semibold">RMSE</div>
              <div>{formatNumber(metrics.rmse, 0)}</div>
            </div>
            <div>
              <div className="text-foreground font-semibold">MAPE</div>
              <div>{formatPct(metrics.mape)}</div>
            </div>
          </div>
        ) : (
          <div className="text-xs text-muted-foreground">
            No feedback yet — add real prices to evaluate accuracy.
          </div>
        )}
      </div>

      {chartData.length > 1 && (
        <div className="mt-6 rounded-xl border border-border bg-background p-4">
          <div className="mb-3 text-sm font-medium text-foreground">Accuracy trend</div>
          <ChartContainer
            config={{
              mae: { label: 'MAE', color: 'hsl(var(--primary))' },
              mape: { label: 'MAPE %', color: 'hsl(var(--destructive))' },
            }}
            className="h-[220px]"
          >
            <LineChart data={chartData}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="bucket" tickLine={false} axisLine={false} minTickGap={20} />
              <YAxis yAxisId="left" tickLine={false} axisLine={false} width={40} />
              <YAxis yAxisId="right" orientation="right" tickLine={false} axisLine={false} width={40} />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    formatter={(value, name) => {
                      if (name === 'mape') return [`${Number(value).toFixed(1)}%`, 'MAPE'];
                      return [Number(value).toFixed(0), 'MAE'];
                    }}
                  />
                }
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="mae"
                stroke="var(--color-mae)"
                strokeWidth={2}
                dot={false}
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="mape"
                stroke="var(--color-mape)"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ChartContainer>
        </div>
      )}

      <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
        <span>Drift score = |recent_mean − baseline_mean| / baseline_std</span>
        <span>Thresholds: Low &lt; 0.5, Medium 0.5–1.0, High ≥ 1.0</span>
        <span>Medium: {mediumCount} • High: {highCount}</span>
      </div>
    </div>
  );
};

export default ModelHealth;
