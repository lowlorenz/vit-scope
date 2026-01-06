/** API client for ViT Scope backend. */

const API_BASE = '/api';

export interface ImageUploadResponse {
  image_id: string;
  grid_h: number;
  grid_w: number;
  patch_size: number;
  img_size: number;
}

export interface CosineSimilarityResult {
  heatmap: number[][];
  grid_h: number;
  grid_w: number;
  selected_patch_index: number;
  layer_index: number;
}

export interface TokenNormResult {
  heatmap: number[][];
  grid_h: number;
  grid_w: number;
  layer_index: number;
  min_value: number;
  max_value: number;
}

export async function uploadImage(file: File, modelName?: string): Promise<ImageUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  if (modelName) {
    formData.append('model_name', modelName);
  }
  
  const response = await fetch(`${API_BASE}/images`, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(error.detail || 'Upload failed');
  }
  
  return response.json();
}

export interface PluginInfo {
  [pluginId: string]: string; // plugin_id -> plugin_name
}

export interface ModelInfo {
  [modelName: string]: string; // model_name -> display_name
}

export async function listPlugins(): Promise<PluginInfo> {
  const response = await fetch(`${API_BASE}/plugins`);
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to fetch plugins' }));
    throw new Error(error.detail || 'Failed to fetch plugins');
  }
  
  return response.json();
}

export async function listModels(): Promise<ModelInfo> {
  const response = await fetch(`${API_BASE}/models`);
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to fetch models' }));
    throw new Error(error.detail || 'Failed to fetch models');
  }
  
  return response.json();
}

export async function selectModel(modelName: string): Promise<{ model_name: string; patch_size: number; img_size: number; message: string }> {
  const response = await fetch(`${API_BASE}/models/select`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ model_name: modelName }),
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to select model' }));
    throw new Error(error.detail || 'Failed to select model');
  }
  
  return response.json();
}

export async function handlePluginEvent(
  pluginId: string,
  imageId: string,
  event: Record<string, any>
): Promise<CosineSimilarityResult | TokenNormResult> {
  const response = await fetch(`${API_BASE}/plugins/${pluginId}/event`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image_id: imageId,
      event,
    }),
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Plugin event failed' }));
    throw new Error(error.detail || 'Plugin event failed');
  }
  
  return response.json();
}


