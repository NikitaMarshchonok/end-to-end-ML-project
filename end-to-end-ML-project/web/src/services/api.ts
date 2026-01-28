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

// Mock data for development/demo
export const mockModels: Model[] = [
  {
    id: 'model-nyc-2024',
    name: 'NYC Residential',
    version: '2.1.0',
    currency: 'USD',
    features: [
      { name: 'sqft', type: 'number', label: 'Square Feet', min: 200, max: 10000, step: 50, unit: 'sq ft', placeholder: '1200' },
      { name: 'bedrooms', type: 'number', label: 'Bedrooms', min: 0, max: 10, step: 1, placeholder: '2' },
      { name: 'bathrooms', type: 'number', label: 'Bathrooms', min: 1, max: 6, step: 0.5, placeholder: '1.5' },
      { name: 'neighborhood', type: 'select', label: 'Neighborhood', options: ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'] },
      { name: 'year_built', type: 'number', label: 'Year Built', min: 1800, max: 2024, step: 1, placeholder: '1985' },
      { name: 'property_type', type: 'select', label: 'Property Type', options: ['Apartment', 'Condo', 'Townhouse', 'Single Family'] },
    ],
  },
  {
    id: 'model-la-2024',
    name: 'LA Metro',
    version: '1.8.3',
    currency: 'USD',
    features: [
      { name: 'sqft', type: 'number', label: 'Square Feet', min: 300, max: 15000, step: 50, unit: 'sq ft', placeholder: '1800' },
      { name: 'bedrooms', type: 'number', label: 'Bedrooms', min: 0, max: 12, step: 1, placeholder: '3' },
      { name: 'bathrooms', type: 'number', label: 'Bathrooms', min: 1, max: 8, step: 0.5, placeholder: '2' },
      { name: 'pool', type: 'select', label: 'Pool', options: ['Yes', 'No'] },
      { name: 'garage_spaces', type: 'number', label: 'Garage Spaces', min: 0, max: 4, step: 1, placeholder: '2' },
      { name: 'lot_size', type: 'number', label: 'Lot Size', min: 1000, max: 50000, step: 100, unit: 'sq ft', placeholder: '6000' },
    ],
  },
];

export const generateMockPrediction = (modelId: string, features: Record<string, string | number>): PredictionResponse => {
  const basePrice = modelId.includes('nyc') ? 850000 : 1200000;
  const sqft = Number(features.sqft) || 1500;
  const bedrooms = Number(features.bedrooms) || 2;
  
  const price = basePrice + (sqft * 350) + (bedrooms * 50000) + Math.random() * 100000;
  const variance = price * 0.15;
  
  return {
    price: Math.round(price),
    p10: Math.round(price - variance),
    p50: Math.round(price),
    p90: Math.round(price + variance),
    model_version: modelId.includes('nyc') ? '2.1.0' : '1.8.3',
    currency: 'USD',
    factors: [
      { name: 'Square Footage', impact: 85, direction: 'up' },
      { name: 'Bedrooms', impact: 65, direction: 'up' },
      { name: 'Location', impact: 72, direction: 'up' },
      { name: 'Building Age', impact: -25, direction: 'down' },
      { name: 'Market Trend', impact: 45, direction: 'up' },
    ],
  };
};
