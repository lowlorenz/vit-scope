import React, { useRef, useEffect, useState } from 'react';
import './PatchGrid.css';

interface PatchGridProps {
  imageUrl: string;
  gridH: number;
  gridW: number;
  patchSize: number;
  imgSize: number;
  onPatchClick: (patchIndex: number, row: number, col: number) => void;
  heatmap?: number[][];
  selectedPatchIndex?: number;
  heatmapMin?: number;
  heatmapMax?: number;
}

export const PatchGrid: React.FC<PatchGridProps> = ({
  imageUrl,
  gridH,
  gridW,
  patchSize,
  imgSize,
  onPatchClick,
  heatmap,
  selectedPatchIndex,
  heatmapMin,
  heatmapMax,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [isFocused, setIsFocused] = useState(false);

  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setContainerSize({ width: rect.width, height: rect.height });
      }
    };

    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  // Keyboard navigation handler
  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (selectedPatchIndex === null || selectedPatchIndex === undefined) return;

    const currentRow = Math.floor(selectedPatchIndex / gridW);
    const currentCol = selectedPatchIndex % gridW;
    let newRow = currentRow;
    let newCol = currentCol;
    let handled = false;

    // Handle hjkl (vim keys) and arrow keys
    switch (e.key) {
      case 'h':
      case 'ArrowLeft':
        // Move left
        if (currentCol > 0) {
          newCol = currentCol - 1;
          handled = true;
        }
        break;
      case 'j':
      case 'ArrowDown':
        // Move down
        if (currentRow < gridH - 1) {
          newRow = currentRow + 1;
          handled = true;
        }
        break;
      case 'k':
      case 'ArrowUp':
        // Move up
        if (currentRow > 0) {
          newRow = currentRow - 1;
          handled = true;
        }
        break;
      case 'l':
      case 'ArrowRight':
        // Move right
        if (currentCol < gridW - 1) {
          newCol = currentCol + 1;
          handled = true;
        }
        break;
    }

    if (handled) {
      e.preventDefault();
      const newPatchIndex = newRow * gridW + newCol;
      handlePatchClick(newPatchIndex, newRow, newCol);
    }
  };

  const handlePatchClick = (patchIndex: number, row: number, col: number) => {
    onPatchClick(patchIndex, row, col);
  };

  const getHeatmapValue = (row: number, col: number): number | null => {
    if (!heatmap || row >= heatmap.length || col >= heatmap[row]?.length) {
      return null;
    }
    return heatmap[row][col];
  };

  const getHeatmapColor = (value: number | null, min?: number, max?: number): string => {
    if (value === null) return 'transparent';
    
    // Use provided min/max or default to -1 to 1 (for cosine similarity)
    const valueMin = min !== undefined ? min : -1;
    const valueMax = max !== undefined ? max : 1;
    
    // Normalize value to 0-1 range
    const normalized = (value - valueMin) / (valueMax - valueMin);
    const clamped = Math.max(0, Math.min(1, normalized)); // Clamp to [0, 1]
    
    // Use a color scale: blue (low) -> green -> yellow -> red (high)
    const hue = (1 - clamped) * 240; // 240 (blue) to 0 (red)
    return `hsla(${hue}, 70%, 50%, 0.6)`;
  };

  const patchWidth = containerSize.width / gridW;
  const patchHeight = containerSize.height / gridH;

  return (
    <div
      className={`patch-grid-container ${isFocused ? 'focused' : ''}`}
      ref={containerRef}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onFocus={() => setIsFocused(true)}
      onBlur={() => setIsFocused(false)}
      onClick={() => containerRef.current?.focus()}
      title={isFocused ? 'Use hjkl or arrow keys to navigate' : 'Click to focus and use keyboard navigation'}
    >
      <img src={imageUrl} alt="Uploaded" className="patch-grid-image" />
      <div className="patch-grid-overlay">
        {Array.from({ length: gridH * gridW }, (_, index) => {
          const row = Math.floor(index / gridW);
          const col = index % gridW;
          const isSelected = selectedPatchIndex === index;
          const heatmapValue = getHeatmapValue(row, col);
          const heatmapColor = getHeatmapColor(heatmapValue, heatmapMin, heatmapMax);

          return (
            <div
              key={index}
              className={`patch-cell ${isSelected ? 'selected' : ''}`}
              style={{
                position: 'absolute',
                left: `${(col / gridW) * 100}%`,
                top: `${(row / gridH) * 100}%`,
                width: `${100 / gridW}%`,
                height: `${100 / gridH}%`,
                backgroundColor: heatmapColor,
                border: isSelected ? '2px solid #007bff' : '1px solid rgba(0, 0, 0, 0.1)',
                cursor: 'pointer',
              }}
              onClick={() => handlePatchClick(index, row, col)}
              title={`Patch ${index} (${row}, ${col})`}
            />
          );
        })}
      </div>
    </div>
  );
};


