const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface Model {
  id: string;
  name: string;
  version: string;
  features: Feature[];
  currency: string;
  market_id: string;
}

export interface Market {
  id: string;
  name: string;
  currency: string;
}

export interface Feature {
  name: string;
  type: 'number' | 'select' | 'text';
  label: string;
  options?: string[];
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  placeholder?: string;
  required?: boolean;
}

export interface PredictionRequest {
  market_id: string;
  model_id: string;
  features: Record<string, string | number>;
  area_unit?: 'm2' | 'sqft';
}

export interface PredictionResponse {
  price: number;
  p10: number;
  p50: number;
  p90: number;
  model_version: string;
  currency: string;
  factors: Factor[];
  prediction_id?: number | null;
  display_currency?: string | null;
  display_price?: number | null;
  display_p10?: number | null;
  display_p50?: number | null;
  display_p90?: number | null;
}

export interface ExplainRange {
  label: string;
  low: number;
  high: number;
}

export interface ExplainResponse {
  model_version: string;
  currency: string;
  factors: Factor[];
  ranges: ExplainRange[];
  display_currency?: string | null;
  display_ranges?: ExplainRange[] | null;
}

export interface PredictionHistoryItem {
  id: number;
  created_at: string;
  market_id: string;
  model_id: string;
  model_version: string;
  currency: string;
  price: number;
  p10: number;
  p50: number;
  p90: number;
  area_unit?: 'm2' | 'sqft';
  features: Record<string, string | number>;
  factors: Factor[];
}

export interface ComparableField {
  key: string;
  label: string;
  unit?: string;
}

export interface ComparableItem {
  price: number;
  distance: number;
  features: Record<string, string | number>;
}

export interface ComparablesResponse {
  currency: string;
  fields: ComparableField[];
  items: ComparableItem[];
}

export interface DriftItem {
  feature: string;
  baseline_mean: number;
  recent_mean: number;
  baseline_std: number;
  drift_score: number;
  sample_size: number;
}

export interface MonitoringResponse {
  model_id: string;
  total_predictions: number;
  sample_size: number;
  drift: DriftItem[];
  note?: string;
}

export interface FeedbackResponse {
  prediction_id: number;
  actual_price: number;
  abs_error: number;
  pct_error: number | null;
  actual_currency?: string;
}

export interface MetricsResponse {
  count: number;
  mae: number | null;
  mape: number | null;
  rmse: number | null;
}

export interface MetricsTimeseriesItem {
  bucket: string;
  count: number;
  mae: number;
  rmse: number;
  mape: number | null;
}

export interface Factor {
  name: string;
  impact: number; // -100 to 100
  direction: 'up' | 'down';
}

export const fetchModels = async (marketId?: string): Promise<Model[]> => {
  const url = marketId ? `${API_BASE_URL}/models?market_id=${marketId}` : `${API_BASE_URL}/models`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error('Failed to fetch models');
  }
  return response.json();
};

export const fetchMarkets = async (): Promise<Market[]> => {
  const response = await fetch(`${API_BASE_URL}/markets`);
  if (!response.ok) {
    throw new Error('Failed to fetch markets');
  }
  return response.json();
};

export const resolveMarket = async (lat: number, long: number): Promise<{ market_id: string; market_name: string; distance_km: number }> => {
  const response = await fetch(`${API_BASE_URL}/markets/resolve?lat=${lat}&long=${long}`);
  if (!response.ok) {
    throw new Error('Failed to resolve market');
  }
  return response.json();
};

export const predictPrice = async (request: PredictionRequest): Promise<PredictionResponse> => {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    throw new Error('Failed to get prediction');
  }
  return response.json();
};

export const explainPrediction = async (request: PredictionRequest): Promise<ExplainResponse> => {
  const response = await fetch(`${API_BASE_URL}/explain`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    throw new Error('Failed to get explanation');
  }
  return response.json();
};

export const fetchPredictions = async (limit = 10): Promise<PredictionHistoryItem[]> => {
  const response = await fetch(`${API_BASE_URL}/predictions?limit=${limit}`);
  if (!response.ok) {
    throw new Error('Failed to fetch predictions history');
  }
  return response.json();
};

export const clearPredictions = async (): Promise<void> => {
  const response = await fetch(`${API_BASE_URL}/predictions`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error('Failed to clear predictions history');
  }
};

export const fetchComparables = async (request: PredictionRequest, topK = 5): Promise<ComparablesResponse> => {
  const response = await fetch(`${API_BASE_URL}/comparables`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ ...request, top_k: topK }),
  });
  if (!response.ok) {
    throw new Error('Failed to fetch comparables');
  }
  return response.json();
};

export const fetchMonitoring = async (modelId: string, marketId?: string): Promise<MonitoringResponse> => {
  const url = marketId
    ? `${API_BASE_URL}/monitoring?model_id=${modelId}&market_id=${marketId}`
    : `${API_BASE_URL}/monitoring?model_id=${modelId}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error('Failed to fetch monitoring data');
  }
  return response.json();
};

export const submitFeedback = async (
  predictionId: number,
  actualPrice: number,
  actualCurrency?: string
): Promise<FeedbackResponse> => {
  const response = await fetch(`${API_BASE_URL}/feedback`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      prediction_id: predictionId,
      actual_price: actualPrice,
      actual_currency: actualCurrency,
    }),
  });
  if (!response.ok) {
    throw new Error('Failed to submit feedback');
  }
  return response.json();
};

export const fetchMetrics = async (modelId?: string, marketId?: string): Promise<MetricsResponse> => {
  const params: string[] = [];
  if (modelId) params.push(`model_id=${modelId}`);
  if (marketId) params.push(`market_id=${marketId}`);
  const url = params.length > 0 ? `${API_BASE_URL}/metrics?${params.join('&')}` : `${API_BASE_URL}/metrics`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error('Failed to fetch metrics');
  }
  return response.json();
};

export const fetchMetricsTimeseries = async (modelId?: string, marketId?: string, bucket: 'day' | 'week' = 'day'): Promise<MetricsTimeseriesItem[]> => {
  const params: string[] = [`bucket=${bucket}`];
  if (modelId) params.push(`model_id=${modelId}`);
  if (marketId) params.push(`market_id=${marketId}`);
  const url = `${API_BASE_URL}/metrics/timeseries?${params.join('&')}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error('Failed to fetch metrics timeseries');
  }
  return response.json();
};

// Mock data for development/demo
export const mockMarkets: Market[] = [
  { id: 'il-tlv', name: 'Israel • Tel Aviv', currency: 'ILS' },
  { id: 'tw-tpe', name: 'Taiwan • Taipei', currency: 'TWD' },
];

export const mockModels: Model[] = [
  {
    id: 'tel_aviv_v3_2_clean',
    name: 'Tel Aviv model v3.2_clean (best)',
    version: '3.2',
    currency: 'ILS',
    market_id: 'il-tlv',
    features: [
      { name: 'netArea', type: 'number', label: 'Net area', min: 10, max: 1000, step: 1, unit: 'm²', placeholder: '80', required: true },
      { name: 'rooms', type: 'number', label: 'Rooms', min: 1, max: 12, step: 0.5, placeholder: '3', required: true },
      { name: 'floor', type: 'number', label: 'Floor', min: -2, max: 200, step: 1, placeholder: '4', required: true },
      { name: 'constructionYear', type: 'number', label: 'Construction year', min: 1900, max: 2100, step: 1, placeholder: '2010', required: true },
      { name: 'grossArea', type: 'number', label: 'Gross area', min: 10, max: 1200, step: 1, unit: 'm²', placeholder: '95', required: false },
      { name: 'floors', type: 'number', label: 'Total floors', min: 1, max: 300, step: 1, placeholder: '12', required: false },
      { name: 'apartmentsInBuilding', type: 'number', label: 'Apartments in building', min: 1, max: 500, step: 1, placeholder: '40', required: false },
      { name: 'parking', type: 'number', label: 'Parking spots', min: 0, max: 5, step: 1, placeholder: '1', required: false },
      { name: 'storage', type: 'number', label: 'Storage', min: 0, max: 3, step: 1, placeholder: '1', required: false },
      { name: 'roof', type: 'number', label: 'Roof area', min: 0, max: 300, step: 1, unit: 'm²', placeholder: '20', required: false },
      { name: 'yard', type: 'number', label: 'Yard area', min: 0, max: 300, step: 1, unit: 'm²', placeholder: '30', required: false },
    ],
  },
  {
    id: 'taiwan',
    name: 'Taiwan tutorial model',
    version: '1.0',
    currency: 'TWD',
    market_id: 'tw-tpe',
    features: [
      { name: 'distance', type: 'number', label: 'Distance to MRT', min: 0, max: 20000, step: 50, unit: 'm', placeholder: '400', required: true },
      { name: 'convenience', type: 'number', label: 'Convenience stores', min: 0, max: 20, step: 1, placeholder: '4', required: true },
      { name: 'lat', type: 'number', label: 'Latitude', min: 20, max: 30, step: 0.0001, placeholder: '24.98', required: true },
      { name: 'long', type: 'number', label: 'Longitude', min: 120, max: 130, step: 0.0001, placeholder: '121.54', required: true },
    ],
  },
];

export const generateMockPrediction = (modelId: string, features: Record<string, string | number>): PredictionResponse => {
  const basePrice = modelId === 'taiwan' ? 30 : 3500000;
  const netArea = Number(features.netArea) || 80;
  const rooms = Number(features.rooms) || 3;
  const distance = Number(features.distance) || 400;

  let price = basePrice;
  if (modelId === 'taiwan') {
    price = basePrice + Math.max(0, (2000 - distance)) * 0.01;
  } else {
    price = basePrice + (netArea * 25000) + (rooms * 150000);
  }

  const variance = price * 0.12;

  return {
    price: Math.round(price),
    p10: Math.round(price - variance),
    p50: Math.round(price),
    p90: Math.round(price + variance),
    model_version: modelId === 'taiwan' ? '1.0' : '3.2',
    currency: modelId === 'taiwan' ? 'TWD' : 'ILS',
    factors: [
      { name: 'Area', impact: 70, direction: 'up' },
      { name: 'Rooms', impact: 55, direction: 'up' },
      { name: 'Floor', impact: 20, direction: 'up' },
      { name: 'Building age', impact: 15, direction: 'down' },
      { name: 'Market trend', impact: 35, direction: 'up' },
    ],
    prediction_id: Math.floor(Math.random() * 10000) + 1,
  };
};

export const generateMockExplain = (modelId: string, features: Record<string, string | number>): ExplainResponse => {
  const prediction = generateMockPrediction(modelId, features);
  return {
    model_version: prediction.model_version,
    currency: prediction.currency,
    factors: prediction.factors,
    ranges: [
      {
        label: 'Typical range (±MAE)',
        low: Math.round(prediction.price * 0.88),
        high: Math.round(prediction.price * 1.12),
      },
      {
        label: 'Conservative range (±RMSE)',
        low: Math.round(prediction.price * 0.80),
        high: Math.round(prediction.price * 1.20),
      },
    ],
  };
};

export const generateMockComparables = (
  modelId: string,
  features: Record<string, string | number>,
  topK = 5
): ComparablesResponse => {
  const currency = modelId === 'taiwan' ? 'TWD' : 'ILS';
  const fields: ComparableField[] =
    modelId === 'taiwan'
      ? [
          { key: 'distance', label: 'Distance to MRT', unit: 'm' },
          { key: 'convenience', label: 'Convenience stores' },
          { key: 'lat', label: 'Latitude' },
          { key: 'long', label: 'Longitude' },
        ]
      : [
          { key: 'netArea', label: 'Net area', unit: 'm²' },
          { key: 'rooms', label: 'Rooms' },
          { key: 'floor', label: 'Floor' },
          { key: 'constructionYear', label: 'Year' },
        ];

  const count = Math.max(1, Math.min(topK, 10));
  const items: ComparableItem[] = Array.from({ length: count }).map((_, idx) => ({
    price: modelId === 'taiwan' ? 30 + idx * 2 : 3000000 + idx * 150000,
    distance: 0.1 + idx * 0.05,
    features: {
      ...features,
      netArea: Number(features.netArea) || 80,
      rooms: Number(features.rooms) || 3,
      floor: Number(features.floor) || 4,
      constructionYear: Number(features.constructionYear) || 2010,
      distance: Number(features.distance) || 400,
      convenience: Number(features.convenience) || 4,
      lat: Number(features.lat) || 25.0,
      long: Number(features.long) || 121.5,
    },
  }));

  return { currency, fields, items };
};

export const generateMockMonitoring = (modelId: string): MonitoringResponse => {
  return {
    model_id: modelId,
    total_predictions: 24,
    sample_size: 24,
    drift: [
      {
        feature: 'netArea',
        baseline_mean: 85,
        recent_mean: 96,
        baseline_std: 32,
        drift_score: 0.34,
        sample_size: 24,
      },
      {
        feature: 'rooms',
        baseline_mean: 3.1,
        recent_mean: 3.4,
        baseline_std: 1.2,
        drift_score: 0.25,
        sample_size: 24,
      },
      {
        feature: 'constructionYear',
        baseline_mean: 1995,
        recent_mean: 2002,
        baseline_std: 18,
        drift_score: 0.39,
        sample_size: 24,
      },
    ],
  };
};

export const generateMockFeedback = (): FeedbackResponse => {
  return {
    prediction_id: Math.floor(Math.random() * 10000) + 1,
    actual_price: 3200000,
    abs_error: 180000,
    pct_error: 0.056,
  };
};

export const generateMockMetrics = (): MetricsResponse => {
  return {
    count: 12,
    mae: 210000,
    mape: 0.07,
    rmse: 320000,
  };
};

export const generateMockMetricsTimeseries = (): MetricsTimeseriesItem[] => {
  return [
    { bucket: '2025-01-05', count: 2, mae: 240000, rmse: 310000, mape: 0.08 },
    { bucket: '2025-01-12', count: 3, mae: 220000, rmse: 300000, mape: 0.075 },
    { bucket: '2025-01-19', count: 4, mae: 205000, rmse: 290000, mape: 0.07 },
    { bucket: '2025-01-26', count: 3, mae: 190000, rmse: 270000, mape: 0.065 },
  ];
};
