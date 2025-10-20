# Start the Process Optimization API

## Quick Start

```bash
cd "d:\Main Process Optimization Agent"
python -m uvicorn API.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### Base URL
```
http://localhost:8000
```

### Available Endpoints

1. **Health Check**
   ```
   GET /health
   ```

2. **Optimize Process (Full Response)**
   ```
   POST /cms/optimize/{process_id}
   ```
   Returns: Complete JSON with optimization results

3. **Get Summary PNG Only**
   ```
   POST /cms/optimize/{process_id}/summary_png
   ```
   Returns: PNG file directly

4. **Get Allocation PNG Only**
   ```
   POST /cms/optimize/{process_id}/alloc_png
   ```
   Returns: PNG file directly

## Configuration

- **CMS URL**: `https://fyp-cms-frontend.vercel.app` (default)
- **Output Directory**: `visualization_outputs/`
- **Optimizer Script**: `test_process_detection.py`

## Testing

Test with process ID 21:
```bash
python quick_test_21.py
```

## Status

✅ **WORKING** - Process ID 21 optimizes successfully
✅ **CMS Connected** - Using deployed CMS
✅ **Visualizations Generated** - Healthcare format
