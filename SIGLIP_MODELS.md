# SigLIP Model Support

This document lists all SigLIP models that are supported by the `SiglipEngine` and any models that are not supported.

## Supported Models

The following **11 SigLIP models** are fully supported and tested:

### SigLIP v1 - Base Models
1. **google/siglip-base-patch16-224**
   - Patch size: 16
   - Image size: 224
   - Hidden size: 768

2. **google/siglip-base-patch16-256**
   - Patch size: 16
   - Image size: 256
   - Hidden size: 768

3. **google/siglip-base-patch16-384**
   - Patch size: 16
   - Image size: 384
   - Hidden size: 768

4. **google/siglip-base-patch16-512**
   - Patch size: 16
   - Image size: 512
   - Hidden size: 768

### SigLIP v1 - Large Models
5. **google/siglip-large-patch16-256**
   - Patch size: 16
   - Image size: 256
   - Hidden size: 1024

6. **google/siglip-large-patch16-384**
   - Patch size: 16
   - Image size: 384
   - Hidden size: 1024

### SigLIP v1 - SoViT Models
7. **google/siglip-so400m-patch14-384** (default)
   - Patch size: 14
   - Image size: 384
   - Hidden size: 1152

### SigLIP 2 - Base Models
8. **google/siglip2-base-patch16-224**
   - Patch size: 16
   - Image size: 224
   - Hidden size: 768

9. **google/siglip2-base-patch16-256**
   - Patch size: 16
   - Image size: 256
   - Hidden size: 768

### SigLIP 2 - Large Models
10. **google/siglip2-large-patch16-256**
    - Patch size: 16
    - Image size: 256
    - Hidden size: 1024

### SigLIP 2 - SoViT Models
11. **google/siglip2-so400m-patch14-384**
    - Patch size: 14
    - Image size: 384
    - Hidden size: 1152
    - Reference: [HuggingFace Model Card](https://huggingface.co/google/siglip2-so400m-patch14-384)

## Unsupported Models

The following models **do not exist** on HuggingFace or are not compatible:

### Non-Existent Models
The following models do not exist on HuggingFace:
- `google/siglip-base-patch32-224` - Does not exist
- `google/siglip-base-patch32-256` - Does not exist
- `google/siglip-base-patch32-384` - Does not exist
- `google/siglip-large-patch32-256` - Does not exist
- `google/siglip-large-patch32-384` - Does not exist
- `google/siglip-large-patch16-512` - Does not exist

**Note**: `google/siglip-base-patch16-512` exists and is supported (see supported models above).

## Testing

All supported models are tested in `backend/tests/test_siglip_models.py`. The test suite includes:

- Model initialization
- Config extraction
- Image normalization
- Forward pass
- Logit lens
- Text embedding (single and batch)
- Hidden states consistency
- Deterministic output

Run tests with:
```bash
cd backend
PYTHONPATH=/data/models/users/hufe/vit-scope/backend uv run --no-project pytest tests/test_siglip_models.py -v
```

## Usage

You can use any supported model by passing the `model_name` parameter:

```python
from app.engine.siglip_engine import SiglipEngine

# Use default model
engine = SiglipEngine()

# Use a different model
engine = SiglipEngine(model_name="google/siglip-base-patch16-224")
```

The engine automatically extracts the correct configuration (patch_size, img_size) from each model.

