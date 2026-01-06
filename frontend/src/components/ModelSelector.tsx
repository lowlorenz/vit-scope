import React from 'react';
import './ModelSelector.css';

interface ModelSelectorProps {
  models: Array<{ name: string; displayName: string }>;
  selectedModel: string;
  onModelChange: (modelName: string) => void;
  disabled?: boolean;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  selectedModel,
  onModelChange,
  disabled = false,
}) => {
  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onModelChange(e.target.value);
  };

  return (
    <div className="model-selector-container">
      <label className="model-selector-label">
        Model:
      </label>
      <select
        value={selectedModel}
        onChange={handleChange}
        disabled={disabled}
        className="model-selector"
      >
        {models.map((model) => (
          <option key={model.name} value={model.name}>
            {model.displayName}
          </option>
        ))}
      </select>
    </div>
  );
};

