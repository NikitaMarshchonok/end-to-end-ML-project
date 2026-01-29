import { ComparablesResponse } from '@/services/api';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Scale } from 'lucide-react';

interface ComparableSalesProps {
  data: ComparablesResponse | null;
}

const formatCurrency = (value: number, currency: string) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    maximumFractionDigits: 0,
  }).format(value);
};

const formatValue = (value: string | number, unit?: string) => {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'number') {
    const rounded = Number.isInteger(value) ? value : Number(value.toFixed(2));
    return unit ? `${rounded} ${unit}` : String(rounded);
  }
  return unit ? `${value} ${unit}` : String(value);
};

const ComparableSales = ({ data }: ComparableSalesProps) => {
  if (!data) {
    return null;
  }

  return (
    <div className="bg-card rounded-2xl shadow-card p-6 border border-border animate-fade-in">
      <div className="flex items-center gap-2 mb-4">
        <div className="p-2 rounded-lg bg-muted">
          <Scale className="h-5 w-5 text-muted-foreground" />
        </div>
        <div>
          <h3 className="font-semibold text-foreground">Comparable Sales</h3>
          <p className="text-sm text-muted-foreground">Top similar properties (lower score = closer)</p>
        </div>
      </div>

      {data.items.length === 0 ? (
        <div className="text-sm text-muted-foreground">
          Not enough input data to find similar listings. Try filling more fields.
        </div>
      ) : (
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Price</TableHead>
            <TableHead>Similarity</TableHead>
            {data.fields.map((field) => (
              <TableHead key={field.key}>{field.label}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.items.map((item, idx) => (
            <TableRow key={`${item.price}-${idx}`}>
              <TableCell className="font-medium">
                {formatCurrency(item.price, data.currency)}
              </TableCell>
              <TableCell className="text-muted-foreground">
                {item.distance.toFixed(2)}
              </TableCell>
              {data.fields.map((field) => (
                <TableCell key={`${item.price}-${field.key}-${idx}`}>
                  {formatValue(item.features[field.key], field.unit)}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
      )}
    </div>
  );
};

export default ComparableSales;
