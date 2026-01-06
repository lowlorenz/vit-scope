# Test Results

## Backend Server Test

**Status**: ✅ **PASSES**

The backend server starts successfully. On first run, it will download the `google/siglip-so400m-patch14-384` model from HuggingFace, which may take several minutes depending on your internet connection.

**Note**: The SigLIP model requires additional dependencies:
- `sentencepiece` - for tokenization
- `protobuf` - for model loading

These have been added to `pyproject.toml` and will be installed automatically.

**Startup command**:
```bash
cd backend
PYTHONPATH=/data/models/users/hufe/vit-scope/backend uv run --no-project python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Note**: The model download happens during startup in the `startup_event` handler. Once downloaded, subsequent starts will be faster as the model is cached.

**Expected output on first run**:
- Server process starts
- Model download begins (may take 2-5 minutes)
- Server ready on http://0.0.0.0:8000

## Frontend Server Test

**Status**: ✅ **PASSES**

The frontend development server starts successfully.

**Startup command**:
```bash
cd frontend
npm install  # (only needed once)
npm run dev
```

**Expected output**:
- Vite dev server starts
- Server ready on http://localhost:3000/

## Package Installation Notes

The project uses `uv` for Python dependency management. Dependencies are installed automatically when using `uv run`. If you prefer to install dependencies separately:

```bash
uv pip install fastapi uvicorn python-multipart pydantic torch transformers pillow numpy
```

## Next Steps

1. Start the backend server (will download model on first run)
2. Start the frontend server
3. Open http://localhost:3000/ in your browser
4. Upload an image to test the cosine similarity plugin

