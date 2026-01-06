# ViT Scope

A tool for efficiently using interpretability methods on Vision Transformers (ViTs). This project uses a plugin-based architecture that makes it easy to extend with new interpretability methods.

## Architecture

### Engine
The engine is used to interact with models. It provides:
- **Forward pass**: Takes normalized input and returns logits and latent activations
- **Normalize**: Takes input and returns normalized input
- **Config**: Provides model configuration (patch_size, img_size)

### Plugins
Plugins are the way to add new utility to the dashboard. They can:
- Compute interpretability methods (e.g., cosine similarity, attention visualization)
- Return heatmaps, HTML code, or other visualizations
- Handle user events (e.g., clicking on a patch)

### Backend
The backend handles calls from the frontend and routes plugin requests. It manages:
- Image uploads and processing
- Activation caching
- Plugin registry and event routing

### Frontend
The frontend provides a web interface for:
- Uploading images (drag and drop)
- Displaying images as patch grids
- Interacting with plugins
- Visualizing interpretability results

## Getting Started

### Backend Setup

1. Install dependencies (using `uv` if available):
```bash
uv pip install -e .
```

2. Start the backend server (from the project root):
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or from the project root with PYTHONPATH:
```bash
PYTHONPATH=backend uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm run dev
```

## Current Model Support

- `google/siglip-so400m-patch14-384`

## Available Plugins

- **Cosine Similarity**: Click on a patch to see cosine similarity with all other patches

## Testing

Run the backend tests:
```bash
cd backend
PYTHONPATH=/data/models/users/hufe/vit-scope/backend uv run --no-project pytest tests/ -v
```

See `backend/tests/README.md` for more details on the test suite.

## For Developers and AI Agents

**Important**: Before contributing, please read **[AGENTS.md](AGENTS.md)** which contains:
- Design principles and coding standards
- Architecture overview
- Testing requirements
- Common pitfalls to avoid
- Guidelines for extending the project

## Extending ViT Scope

### Adding a New Plugin

1. Create a new file in `backend/app/plugins/` (e.g., `my_plugin.py`)
2. Implement the `Plugin` interface from `backend/app/plugins/base.py`
3. Register your plugin in `backend/app/plugins/registry.py`
4. Add tests in `backend/tests/test_plugins.py`

### Adding a New Model

1. Create a new engine class in `backend/app/engine/` implementing the `Engine` interface
2. Update the engine factory to support your model
3. Add tests in `backend/tests/test_engine.py`

