import React from 'react';
import './PluginSidebar.css';

export interface Plugin {
  id: string;
  name: string;
}

interface PluginSidebarProps {
  plugins: Plugin[];
  selectedPluginId: string | null;
  onSelectPlugin: (pluginId: string) => void;
  loading?: boolean;
}

export const PluginSidebar: React.FC<PluginSidebarProps> = ({
  plugins,
  selectedPluginId,
  onSelectPlugin,
  loading = false,
}) => {
  return (
    <div className="plugin-sidebar">
      <div className="plugin-sidebar-header">
        <h2>Plugins</h2>
      </div>
      <div className="plugin-sidebar-content">
        {loading ? (
          <div className="plugin-loading">Loading plugins...</div>
        ) : plugins.length === 0 ? (
          <div className="plugin-empty">No plugins available</div>
        ) : (
          <ul className="plugin-list">
            {plugins.map((plugin) => (
              <li key={plugin.id}>
                <button
                  className={`plugin-item ${
                    selectedPluginId === plugin.id ? 'active' : ''
                  }`}
                  onClick={() => onSelectPlugin(plugin.id)}
                  disabled={loading}
                >
                  <span className="plugin-name">{plugin.name}</span>
                  {selectedPluginId === plugin.id && (
                    <span className="plugin-check">✓</span>
                  )}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

