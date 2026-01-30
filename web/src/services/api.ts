const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface Model {
  id: string;
  name: string;
  version: string;
  features: Feature[];
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
  model_id: string;
  features: Record<string, string | number>;
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
}

export interface PredictionHistoryItem {
  id: number;
  created_at: string;
  model_id: string;
  model_version: string;
  currency: string;
  price: number;
  p10: number;
  p50: number;
  p90: number;
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
}

export interface MetricsResponse {
  count: number;
  mae: number | null;
  mape: number | null;
  rmse: number | null;
}

export interface Factor {
  name: string;
  impact: number; // -100 to 100
  direction: 'up' | 'down';
}

export const fetchModels = async (): Promise<Model[]> => {
  const response = await fetch(`${API_BASE_URL}/models`);
  if (!response.ok) {
    throw new Error('Failed to fetch models');
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

export const fetchMonitoring = async (modelId: string): Promise<MonitoringResponse> => {
  const response = await fetch(`${API_BASE_URL}/monitoring?model_id=${modelId}`);
  if (!response.ok) {
    throw new Error('Failed to fetch monitoring data');
  }
  return response.json();
};

export const submitFeedback = async (predictionId: number, actualPrice: number): Promise<FeedbackResponse> => {
  const response = await fetch(`${API_BASE_URL}/feedback`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      prediction_id: predictionId,
      actual_price: actualPrice,
    }),
  });
  if (!response.ok) {
    throw new Error('Failed to submit feedback');
  }
  return response.json();
};

export const fetchMetrics = async (modelId?: string): Promise<MetricsResponse> => {
  const url = modelId ? `${API_BASE_URL}/metrics?model_id=${modelId}` : `${API_BASE_URL}/metrics`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error('Failed to fetch metrics');
  }
  return response.json();
};

// Mock data for development/demo
export const mockModels: Model[] = [
  {
    id: 'tel_aviv_v3_2_clean',
    name: 'Tel Aviv model v3.2_clean (best)',
    version: '3.2',
    currency: 'ILS',
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
