import React, { useState, useEffect } from 'react';
import { ImageDropzone } from './components/ImageDropzone';
import { PatchGrid } from './components/PatchGrid';
import { PluginSidebar } from './components/PluginSidebar';
import { LayerSlider } from './components/LayerSlider';
import { Colorbar } from './components/Colorbar';
import { ModelSelector } from './components/ModelSelector';
import { uploadImage, handlePluginEvent, listPlugins, listModels, selectModel } from './api/client';
import type { ImageUploadResponse, CosineSimilarityResult, TokenNormResult, PluginInfo, ModelInfo } from './api/client';
import './App.css';

function App() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageData, setImageData] = useState<ImageUploadResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [heatmap, setHeatmap] = useState<number[][] | null>(null);
  const [selectedPatchIndex, setSelectedPatchIndex] = useState<number | null>(null);
  const [plugins, setPlugins] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedPluginId, setSelectedPluginId] = useState<string | null>(null);
  const [pluginsLoading, setPluginsLoading] = useState(true);
  const [selectedLayer, setSelectedLayer] = useState<number>(27); // Default to final layer (27 = last encoder layer)
  const [numLayers] = useState<number>(28); // Default: 1 embedding + 27 layers
  const [heatmapMin, setHeatmapMin] = useState<number>(-1);
  const [heatmapMax, setHeatmapMax] = useState<number>(1);
  const [heatmapLabel, setHeatmapLabel] = useState<string>('Cosine Similarity');
  const [models, setModels] = useState<Array<{ name: string; displayName: string }>>([]);
  const [selectedModel, setSelectedModel] = useState<string>('google/siglip-so400m-patch14-384');
  const [modelsLoading, setModelsLoading] = useState(true);

  // Load plugins and models on mount
  useEffect(() => {
    const loadPlugins = async () => {
      try {
        setPluginsLoading(true);
        const pluginInfo: PluginInfo = await listPlugins();
        const pluginList = Object.entries(pluginInfo).map(([id, name]) => ({ id, name }));
        setPlugins(pluginList);
        
        // Auto-select first plugin if available
        if (pluginList.length > 0 && !selectedPluginId) {
          setSelectedPluginId(pluginList[0].id);
        }
      } catch (err) {
        console.error('Failed to load plugins:', err);
        setError(err instanceof Error ? err.message : 'Failed to load plugins');
      } finally {
        setPluginsLoading(false);
      }
    };

    const loadModels = async () => {
      try {
        setModelsLoading(true);
        const modelInfo: ModelInfo = await listModels();
        const modelList = Object.entries(modelInfo).map(([name, displayName]) => ({ name, displayName }));
        setModels(modelList);
        
        // Set default model if available
        if (modelList.length > 0) {
          const defaultModel = modelList.find(m => m.name === 'google/siglip-so400m-patch14-384');
          if (defaultModel) {
            setSelectedModel(defaultModel.name);
          } else {
            setSelectedModel(modelList[0].name);
          }
        }
      } catch (err) {
        console.error('Failed to load models:', err);
        setError(err instanceof Error ? err.message : 'Failed to load models');
      } finally {
        setModelsLoading(false);
      }
    };

    loadPlugins();
    loadModels();
  }, []);

  // Handle plugin changes: clear heatmap and recompute if needed
  useEffect(() => {
    if (!imageData || !selectedPluginId) return;

    // Clear heatmap when plugin changes
    setHeatmap(null);

    if (selectedPluginId === 'token_norm') {
      // Token norm: always recompute when plugin is selected
      computeTokenNorm(imageData.image_id);
    } else if (selectedPluginId === 'cosine_tokens' && selectedPatchIndex !== null) {
      // Cosine similarity: recompute if a patch is already selected
      // Don't call handlePatchClick as it would update selectedPatchIndex
      // Instead, directly compute the cosine similarity
      const recomputeCosineSimilarity = async () => {
        setLoading(true);
        setError(null);
        try {
          const eventData = {
            type: 'patch_click',
            patch_index: selectedPatchIndex,
            layer_index: selectedLayer,
          };
          
          const result = await handlePluginEvent(
            selectedPluginId,
            imageData.image_id,
            eventData
          );

          if ('selected_patch_index' in result) {
            const cosineResult = result as CosineSimilarityResult;
            setHeatmap(cosineResult.heatmap);
            setHeatmapMin(-1);
            setHeatmapMax(1);
            setHeatmapLabel('Cosine Similarity');
            if (cosineResult.layer_index !== undefined) {
              setSelectedLayer(cosineResult.layer_index);
            }
          }
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to recompute cosine similarity');
        } finally {
          setLoading(false);
        }
      };
      recomputeCosineSimilarity();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPluginId]);

  const computeTokenNorm = async (imageId: string) => {
    if (!selectedPluginId || selectedPluginId !== 'token_norm') return;

    setLoading(true);
    setError(null);

    try {
      const result = await handlePluginEvent(
        selectedPluginId,
        imageId,
        {
          type: 'compute_norm',
          layer_index: selectedLayer,
        }
      );

      // Type guard to check if it's a TokenNormResult
      if ('min_value' in result && 'max_value' in result) {
        const normResult = result as TokenNormResult;
        setHeatmap(normResult.heatmap);
        setHeatmapMin(normResult.min_value);
        setHeatmapMax(normResult.max_value);
        setHeatmapLabel('Token Norm');
        if (normResult.layer_index !== undefined) {
          setSelectedLayer(normResult.layer_index);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to compute token norm');
    } finally {
      setLoading(false);
    }
  };

  const handleModelChange = async (modelName: string) => {
    setSelectedModel(modelName);
    setLoading(true);
    setError(null);

    try {
      // Switch model on backend
      await selectModel(modelName);
      
      // Clear current image data if model changes
      setImageUrl(null);
      setImageData(null);
      setHeatmap(null);
      setSelectedPatchIndex(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to switch model');
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = async (file: File) => {
    setLoading(true);
    setError(null);
    setHeatmap(null);
    setSelectedPatchIndex(null);

    try {
      // Create object URL for display
      const url = URL.createObjectURL(file);
      setImageUrl(url);

      // Upload to backend with selected model
      const response = await uploadImage(file, selectedModel);
      setImageData(response);
      
      // Reset layer to final layer when new image is uploaded
      // Calculate num_layers from the model (will be updated after image loads)
      setSelectedLayer(27); // Default, will be updated
      
      // Auto-compute token norm if that plugin is selected
      if (selectedPluginId === 'token_norm') {
        await computeTokenNorm(response.image_id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload image');
      setImageUrl(null);
    } finally {
      setLoading(false);
    }
  };

  const handlePatchClick = async (patchIndex: number, _row: number, _col: number) => {
    if (!imageData) return;

    // Always update the selected patch (viewer state is independent of plugin)
    setSelectedPatchIndex(patchIndex);

    // Only trigger plugin computation if a plugin is selected and it handles patch clicks
    if (!selectedPluginId) return;

    // Check if the plugin handles patch_click events
    // For now, cosine_tokens handles patch clicks, token_norm doesn't
    // In the future, plugins can declare what events they handle
    if (selectedPluginId !== 'cosine_tokens') {
      // Plugin doesn't handle patch clicks, just update selection
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const eventData = {
        type: 'patch_click',
        patch_index: patchIndex,
        layer_index: selectedLayer,
      };
      
      const result = await handlePluginEvent(
        selectedPluginId,
        imageData.image_id,
        eventData
      );

      // Type guard to check if it's a CosineSimilarityResult
      if ('selected_patch_index' in result) {
        const cosineResult = result as CosineSimilarityResult;
        setHeatmap(cosineResult.heatmap);
        setHeatmapMin(-1);
        setHeatmapMax(1);
        setHeatmapLabel('Cosine Similarity');
        // Update selected layer to match what was actually used
        if (cosineResult.layer_index !== undefined) {
          setSelectedLayer(cosineResult.layer_index);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to compute plugin result');
    } finally {
      setLoading(false);
    }
  };

  const handleLayerChange = async (newLayer: number) => {
    setSelectedLayer(newLayer);
    
    if (!imageData || !selectedPluginId) return;
    
    setLoading(true);
    setError(null);

    try {
      if (selectedPluginId === 'token_norm') {
        // Token norm plugin: recompute with new layer
        const result = await handlePluginEvent(
          selectedPluginId,
          imageData.image_id,
          {
            type: 'compute_norm',
            layer_index: newLayer,
          }
        );

        // Type guard to check if it's a TokenNormResult
        if ('min_value' in result && 'max_value' in result) {
          const normResult = result as TokenNormResult;
          setHeatmap(normResult.heatmap);
          setHeatmapMin(normResult.min_value);
          setHeatmapMax(normResult.max_value);
          if (normResult.layer_index !== undefined) {
            setSelectedLayer(normResult.layer_index);
          }
        }
      } else if (selectedPatchIndex !== null) {
        // Cosine similarity: recompute with new layer if patch is selected
        const eventData = {
          type: 'patch_click',
          patch_index: selectedPatchIndex,
          layer_index: newLayer,
        };
        
        const result = await handlePluginEvent(
          selectedPluginId,
          imageData.image_id,
          eventData
        );

        // Type guard to check if it's a CosineSimilarityResult
        if ('selected_patch_index' in result) {
          const cosineResult = result as CosineSimilarityResult;
          setHeatmap(cosineResult.heatmap);
          setHeatmapMin(-1);
          setHeatmapMax(1);
          setHeatmapLabel('Cosine Similarity');
          if (cosineResult.layer_index !== undefined) {
            setSelectedLayer(cosineResult.layer_index);
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to recompute with new layer');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>ViT Scope</h1>
        <p>Interpretability tools for Vision Transformers</p>
      </header>

      <div className="app-body">
        <PluginSidebar
          plugins={plugins}
          selectedPluginId={selectedPluginId}
          onSelectPlugin={setSelectedPluginId}
          loading={pluginsLoading}
        />
        <main className="app-main">
          {!imageUrl ? (
            <div className="upload-section">
              <div className="model-selection-section">
                <ModelSelector
                  models={models}
                  selectedModel={selectedModel}
                  onModelChange={handleModelChange}
                  disabled={loading || modelsLoading}
                />
              </div>
              <ImageDropzone onImageUpload={handleImageUpload} disabled={loading} />
              {error && <div className="error-message">{error}</div>}
            </div>
          ) : (
            <div className="viewer-section">
              <div className="viewer-content">
                {imageData && (
                  <PatchGrid
                    imageUrl={imageUrl}
                    gridH={imageData.grid_h}
                    gridW={imageData.grid_w}
                    patchSize={imageData.patch_size}
                    imgSize={imageData.img_size}
                    onPatchClick={handlePatchClick}
                    heatmap={heatmap || undefined}
                    selectedPatchIndex={selectedPatchIndex || undefined}
                    heatmapMin={heatmapMin}
                    heatmapMax={heatmapMax}
                  />
                )}
                {heatmap && (selectedPluginId === 'cosine_tokens' || selectedPluginId === 'token_norm') && (
                  <Colorbar min={heatmapMin} max={heatmapMax} label={heatmapLabel} />
                )}
              </div>
              {loading && <div className="loading-overlay">Processing...</div>}
              {error && <div className="error-message">{error}</div>}
              <div className="controls">
                {(selectedPluginId === 'cosine_tokens' || selectedPluginId === 'token_norm') && imageData && (
                  <LayerSlider
                    numLayers={numLayers}
                    selectedLayer={selectedLayer}
                    onLayerChange={handleLayerChange}
                    disabled={loading}
                  />
                )}
                <button
                  onClick={() => {
                    setImageUrl(null);
                    setImageData(null);
                    setHeatmap(null);
                    setSelectedPatchIndex(null);
                    setError(null);
                    setSelectedLayer(27);
                  }}
                  className="reset-button"
                >
                  Upload New Image
                </button>
                {selectedPatchIndex !== null && (
                  <div className="info-text">
                    Selected patch: {selectedPatchIndex}
                    {selectedPluginId && (
                      <>
                        {' '}(using {plugins.find((p) => p.id === selectedPluginId)?.name || selectedPluginId})
                        {(selectedPluginId === 'cosine_tokens' || selectedPluginId === 'token_norm') && (
                          <span> - Layer {selectedLayer === 0 ? 'Embeddings' : selectedLayer - 1}</span>
                        )}
                      </>
                    )}
                  </div>
                )}
                {selectedPluginId === 'token_norm' && selectedPatchIndex === null && (
                  <div className="info-text">
                    Showing token norms (using{' '}
                    {plugins.find((p) => p.id === selectedPluginId)?.name || selectedPluginId})
                    <span> - Layer {selectedLayer === 0 ? 'Embeddings' : selectedLayer - 1}</span>
                  </div>
                )}
                {!selectedPluginId && (
                  <div className="info-text warning">
                    Please select a plugin from the sidebar
                  </div>
                )}
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;


