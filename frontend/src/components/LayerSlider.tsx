import React, { useEffect } from 'react';
import './LayerSlider.css';

interface LayerSliderProps {
  numLayers: number;
  selectedLayer: number;
  onLayerChange: (layer: number) => void;
  disabled?: boolean;
}

export const LayerSlider: React.FC<LayerSliderProps> = ({
  numLayers,
  selectedLayer,
  onLayerChange,
  disabled = false,
}) => {
  const sliderRef = React.useRef<HTMLInputElement>(null);
  const containerRef = React.useRef<HTMLDivElement>(null);
  const wasFocusedRef = React.useRef<boolean>(false);

  // Restore focus after layer changes or when re-enabled (e.g., after async recomputation)
  useEffect(() => {
    if (wasFocusedRef.current && sliderRef.current && !disabled) {
      // Use requestAnimationFrame to ensure focus happens after render
      const frameId = requestAnimationFrame(() => {
        // Double-check the element still exists and isn't disabled
        if (sliderRef.current && !sliderRef.current.disabled) {
          sliderRef.current.focus();
        }
      });
      return () => cancelAnimationFrame(frameId);
    }
  }, [selectedLayer, disabled]);

  // Also restore focus when transitioning from disabled to enabled
  const prevDisabledRef = React.useRef(disabled);
  useEffect(() => {
    if (prevDisabledRef.current && !disabled && wasFocusedRef.current && sliderRef.current) {
      // Was disabled, now enabled - restore focus
      const frameId = requestAnimationFrame(() => {
        if (sliderRef.current && !sliderRef.current.disabled) {
          sliderRef.current.focus();
        }
      });
      prevDisabledRef.current = disabled;
      return () => cancelAnimationFrame(frameId);
    }
    prevDisabledRef.current = disabled;
  }, [disabled]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newLayer = parseInt(e.target.value, 10);
    onLayerChange(newLayer);
  };

  // Track focus state
  const handleFocus = () => {
    wasFocusedRef.current = true;
  };

  const handleBlur = () => {
    wasFocusedRef.current = false;
  };

  // Handle container click/focus - forward to slider
  const handleContainerClick = () => {
    if (!disabled && sliderRef.current) {
      sliderRef.current.focus();
    }
  };

  const handleContainerFocus = () => {
    if (!disabled && sliderRef.current) {
      sliderRef.current.focus();
    }
  };

  // Keyboard navigation handler
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    let newLayer = selectedLayer;
    let handled = false;

    switch (e.key) {
      case 'ArrowLeft':
      case 'ArrowDown':
        // Decrease layer
        if (selectedLayer > 0) {
          newLayer = selectedLayer - 1;
          handled = true;
        }
        break;
      case 'ArrowRight':
      case 'ArrowUp':
        // Increase layer
        if (selectedLayer < numLayers - 1) {
          newLayer = selectedLayer + 1;
          handled = true;
        }
        break;
      case 'Home':
        // Go to first layer
        newLayer = 0;
        handled = true;
        break;
      case 'End':
        // Go to last layer
        newLayer = numLayers - 1;
        handled = true;
        break;
      case 'PageDown':
        // Decrease by 5
        newLayer = Math.max(0, selectedLayer - 5);
        handled = true;
        break;
      case 'PageUp':
        // Increase by 5
        newLayer = Math.min(numLayers - 1, selectedLayer + 5);
        handled = true;
        break;
    }

    if (handled) {
      e.preventDefault();
      e.stopPropagation();
      // Mark that we want to keep focus
      wasFocusedRef.current = true;
      onLayerChange(newLayer);
    }
  };

  // Layer labels: 0 = embeddings, 1-27 = encoder layers
  const getLayerLabel = (layerIndex: number): string => {
    if (layerIndex === 0) {
      return 'Embeddings';
    }
    return `Layer ${layerIndex - 1}`;
  };

  return (
    <div
      ref={containerRef}
      className="layer-slider-container"
      tabIndex={disabled ? -1 : 0}
      onClick={handleContainerClick}
      onFocus={handleContainerFocus}
      title="Click to focus, then use arrow keys, PageUp/PageDown, or Home/End to navigate layers"
    >
      <label className="layer-slider-label">
        Layer: <span className="layer-slider-value">{getLayerLabel(selectedLayer)}</span>
      </label>
      <input
        ref={sliderRef}
        type="range"
        min="0"
        max={numLayers - 1}
        value={selectedLayer}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        onFocus={handleFocus}
        onBlur={handleBlur}
        disabled={disabled}
        className="layer-slider"
        title="Use arrow keys, PageUp/PageDown, or Home/End to navigate layers"
      />
      <div className="layer-slider-info">
        <span className="layer-slider-min">{getLayerLabel(0)}</span>
        <span className="layer-slider-max">{getLayerLabel(numLayers - 1)}</span>
      </div>
    </div>
  );
};

