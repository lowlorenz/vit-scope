import React from 'react';
import './Colorbar.css';

interface ColorbarProps {
  min?: number;
  max?: number;
  label?: string;
}

export const Colorbar: React.FC<ColorbarProps> = ({
  min = -1,
  max = 1,
  label = 'Cosine Similarity',
}) => {
  // Generate gradient stops matching the heatmap color scale
  // Same logic as PatchGrid: blue (low) -> green -> yellow -> red (high)
  const getColorForValue = (value: number): string => {
    // Normalize value to 0-1 range
    const normalized = (value - min) / (max - min);
    // Use a color scale: blue (low) -> green -> yellow -> red (high)
    const hue = (1 - normalized) * 240; // 240 (blue) to 0 (red)
    return `hsl(${hue}, 70%, 50%)`;
  };

  // Create gradient stops
  const numStops = 100;
  const gradientStops: string[] = [];
  for (let i = 0; i <= numStops; i++) {
    const value = min + (max - min) * (i / numStops);
    const color = getColorForValue(value);
    const position = (i / numStops) * 100;
    gradientStops.push(`${color} ${position}%`);
  }

  const gradient = `linear-gradient(to top, ${gradientStops.join(', ')})`;

  return (
    <div className="colorbar-container">
      <div className="colorbar-label">{label}</div>
      <div className="colorbar-gradient" style={{ background: gradient }}>
        <div className="colorbar-ticks">
          <div className="colorbar-tick" style={{ bottom: '0%' }}>
            <span className="colorbar-tick-line"></span>
            <span className="colorbar-tick-label">{min.toFixed(1)}</span>
          </div>
          <div className="colorbar-tick" style={{ bottom: '50%' }}>
            <span className="colorbar-tick-line"></span>
            <span className="colorbar-tick-label">{((min + max) / 2).toFixed(1)}</span>
          </div>
          <div className="colorbar-tick" style={{ bottom: '100%' }}>
            <span className="colorbar-tick-line"></span>
            <span className="colorbar-tick-label">{max.toFixed(1)}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

