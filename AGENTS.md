# AGENTS.md - Development Guidelines for ViT Scope

This document provides essential information for AI agents and developers working on ViT Scope. It outlines design principles, project scope, architecture, and quality standards.

## Project Overview

**ViT Scope** is a tool for efficiently using interpretability methods on Vision Transformers (ViTs). The project uses a plugin-based architecture that makes it easy to extend with new interpretability methods and support new models.

### Core Goals

1. **General Purpose**: Support multiple ViT models, not just one specific model
2. **Extensible**: Plugin-based architecture for easy addition of new interpretability methods
3. **Clean Architecture**: Clear separation between engine, plugins, backend, and frontend
4. **Well Tested**: Comprehensive test coverage for all components
5. **Production Ready**: Code quality standards enforced throughout

## Design Principles

### 1. **Always Implement Interfaces Properly**

**CRITICAL**: Never return dummy values, placeholders, or "not implemented" stubs. If an interface requires a return value, implement it correctly.

- ✅ **DO**: Return actual model outputs, even if they're embeddings rather than classification logits
- ❌ **DON'T**: Return `dummy_logits`, `None`, or arbitrary slices like `[:, :100]`

**Example**: For SigLIP (a contrastive model), the `forward()` method returns the actual pooled embedding as "logits" because that's what the model produces. This is semantically correct even though it's not classification logits.

### 2. **Separation of Concerns**

- **Engine**: Model interaction only. Handles forward passes, normalization, text embedding, etc.
- **Plugins**: Interpretability computations. Should not depend on specific model implementations.
- **Backend**: API layer, routing, caching. Should not contain business logic.
- **Frontend**: UI and user interaction. Should be independent of plugin implementations.

### 3. **Viewer State Independence**

The viewer (patch selection, layer selection) is **independent** of plugins:
- Patch selection persists across plugin changes
- Layer selection is a viewer feature, not a plugin feature
- Plugins can use viewer state but don't control it

### 4. **Plugin Independence**

Plugins should be independent of each other:
- No plugin-to-plugin dependencies
- Plugins communicate only through the plugin interface
- Each plugin handles its own state

### 5. **Type Safety and Documentation**

- Use proper type hints throughout (Python 3.12+ syntax)
- Document all public methods and interfaces
- Include shape information in docstrings for tensors
- Use Pydantic models for API contracts

## Project Scope

### In Scope

- **Vision Transformer Models**: Support for ViT-based models (SigLIP, CLIP, etc.)
- **Interpretability Methods**: Plugins for various interpretability techniques
  - Token-level analysis (cosine similarity, norms, etc.)
  - Layer-wise analysis
  - Attention visualization (future)
  - Activation analysis (future)
- **Web Interface**: Modern React/TypeScript frontend
- **Extensibility**: Easy plugin and model addition

### Out of Scope (Currently)

- Database-backed activation storage (planned for future)
- Multi-image batch processing
- Model training or fine-tuning
- Non-ViT architectures (CNNs, etc.)

## Architecture

### Engine Layer (`backend/app/engine/`)

**Purpose**: Abstract interface for model interaction

**Key Components**:
- `base.py`: Abstract `Engine` interface
- `siglip_engine.py`: SigLIP model implementation

**Interface Requirements**:
- `config`: Returns `EngineConfig` (patch_size, img_size)
- `normalize(image)`: Normalizes input for the model
- `forward(normalized_input)`: Returns `(logits, activations)`
  - `logits`: Final model output (properly implemented, not dummy)
  - `activations`: Dict with `token_embeddings` and `hidden_states`
- `logit_lens(residual_stream)`: Applies classification head to any layer
- `embed_text(text)`: Embeds text input (for multimodal models)

**Important**: All methods must be properly implemented. No placeholders.

### Plugin Layer (`backend/app/plugins/`)

**Purpose**: Interpretability method implementations

**Key Components**:
- `base.py`: Abstract `Plugin` interface
- `registry.py`: Plugin discovery and registration
- Individual plugin files (e.g., `cosine_tokens.py`, `token_norm.py`)

**Plugin Interface**:
- `plugin_id`: Unique identifier
- `name`: Human-readable name
- `handle_event(image_id, event, activations, engine_config)`: Processes events

**Event Types**:
- `patch_click`: User clicked on a patch (includes `patch_index`, optional `layer_index`)
- `compute_norm`: Compute token norms (includes optional `layer_index`)
- Future event types can be added as needed

### Backend API (`backend/app/api/`)

**Purpose**: REST API for frontend communication

**Key Routes**:
- `POST /api/images`: Upload and process image
- `GET /api/plugins`: List available plugins
- `POST /api/plugins/{plugin_id}/event`: Handle plugin events

**Responsibilities**:
- Image processing and caching
- Plugin event routing
- Activation storage

### Frontend (`frontend/src/`)

**Purpose**: User interface for interacting with interpretability tools

**Key Components**:
- `App.tsx`: Main application component
- `components/`: Reusable UI components
  - `PatchGrid`: Image viewer with patch overlay
  - `LayerSlider`: Layer selection control
  - `Colorbar`: Heatmap color scale
  - `PluginSidebar`: Plugin selection
- `api/client.ts`: API client for backend communication

**State Management**:
- Viewer state (selected patch, selected layer) is independent of plugins
- Plugin state (heatmaps, results) is managed per plugin
- State persists across plugin changes

## Code Quality Standards

### Testing

**Requirement**: All new features must include tests.

**Test Structure**:
- `tests/test_engine.py`: Engine tests
- `tests/test_plugins.py`: Plugin tests
- `tests/test_api.py`: API endpoint tests
- `tests/test_cache.py`: Cache tests

**Test Guidelines**:
- Use pytest fixtures for common setup
- Test both success and error cases
- Test edge cases (empty inputs, boundary conditions)
- Use descriptive test names
- Keep tests focused and independent

**Running Tests**:
```bash
cd backend
PYTHONPATH=/data/models/users/hufe/vit-scope/backend uv run --no-project pytest tests/ -v
```

### Code Style

**Python**:
- Follow PEP 8
- Use type hints (Python 3.12+ syntax)
- Document all public methods
- Use descriptive variable names

**TypeScript**:
- Use TypeScript strict mode
- Define interfaces for all data structures
- Avoid `any` types
- Use functional components with hooks

### Error Handling

- Use appropriate HTTP status codes in API
- Provide clear error messages
- Handle edge cases gracefully
- Log errors appropriately (don't expose internals to users)

### Documentation

- Update README.md for user-facing changes
- Update this file (AGENTS.md) for architectural changes
- Document all public APIs
- Include examples in docstrings

## Development Workflow

### Adding a New Plugin

1. **Create Plugin File** (`backend/app/plugins/my_plugin.py`):
   ```python
   from .base import Plugin
   
   class MyPlugin(Plugin):
       @property
       def plugin_id(self) -> str:
           return "my_plugin"
       
       @property
       def name(self) -> str:
           return "My Plugin"
       
       def handle_event(self, image_id, event, activations, engine_config):
           # Implementation
           pass
   ```

2. **Register Plugin** (`backend/app/plugins/registry.py`):
   ```python
   from .my_plugin import MyPlugin
   
   def _register_builtin_plugins(self):
       self.register(MyPlugin())
   ```

3. **Add Tests** (`backend/tests/test_plugins.py`):
   - Test plugin properties
   - Test event handling
   - Test edge cases

4. **Update Frontend** (if needed):
   - Add plugin-specific UI components
   - Handle plugin-specific events

### Adding a New Model

1. **Create Engine** (`backend/app/engine/my_model_engine.py`):
   - Implement all abstract methods from `Engine`
   - **Properly implement `forward()`** - return actual logits, not dummies
   - Implement `logit_lens()` correctly
   - Implement `embed_text()` if multimodal

2. **Add Tests** (`backend/tests/test_engine.py`):
   - Test all interface methods
   - Test output shapes
   - Test device handling

3. **Update Factory** (if needed):
   - Add model selection logic
   - Update configuration

## Common Pitfalls to Avoid

### ❌ Don't Return Dummy Values

```python
# BAD
def forward(self, input):
    return torch.zeros(1, 100), activations  # Dummy logits

# GOOD
def forward(self, input):
    logits = self.model.compute_actual_output(input)
    return logits, activations
```

### ❌ Don't Mix Concerns

```python
# BAD - Plugin accessing model directly
class MyPlugin:
    def handle_event(self, ...):
        model = SiglipEngine()  # Don't do this

# GOOD - Plugin uses activations provided
class MyPlugin:
    def handle_event(self, image_id, event, activations, ...):
        tokens = activations["token_embeddings"]  # Use provided data
```

### ❌ Don't Break Interface Contracts

```python
# BAD - Returns wrong type
def forward(self, input):
    return None, activations  # Interface requires Tensor

# GOOD - Returns correct type
def forward(self, input):
    logits = self.compute_logits(input)
    return logits, activations  # Correct type
```

### ❌ Don't Skip Tests

Every feature must have tests. No exceptions.

## Future Directions

### Planned Features

1. **Database Backend**: Persistent activation storage
2. **More Plugins**: Attention visualization, activation patching, etc.
3. **More Models**: CLIP, DINOv2, etc.
4. **Batch Processing**: Multiple images at once
5. **Export Features**: Save visualizations, export activations

### Extension Points

The architecture is designed to be extended:
- **New Plugins**: Add to `plugins/` directory
- **New Models**: Implement `Engine` interface
- **New Event Types**: Extend plugin interface
- **New UI Components**: Add to `frontend/src/components/`

## Questions?

If you're unsure about:
- **Interface implementation**: Check existing implementations (e.g., `siglip_engine.py`)
- **Plugin structure**: Look at existing plugins (`cosine_tokens.py`, `token_norm.py`)
- **Testing patterns**: Review `tests/test_engine.py` or `tests/test_plugins.py`
- **API patterns**: Check `backend/app/api/routes.py`

## Summary

**Key Takeaways**:
1. ✅ Always implement interfaces properly - no dummies or placeholders
2. ✅ Keep concerns separated - engine, plugins, backend, frontend
3. ✅ Write tests for everything
4. ✅ Keep viewer state independent of plugins
5. ✅ Document your code
6. ✅ Follow type safety and error handling best practices

**Remember**: This is a tool that will be extended by many people. Clean, tested, properly implemented code is essential for maintainability and extensibility.

