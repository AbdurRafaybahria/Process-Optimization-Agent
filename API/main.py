
import os
import sys
import json
import glob
import base64
import asyncio
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from datetime import datetime
import tempfile

# Ensure project root is on sys.path so we can import run_rl_optimizer
PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import webbrowser as _webbrowser
import os as _os

try:
    import run_rl_optimizer as optimizer
except Exception as e:
    raise RuntimeError(f"Failed to import run_rl_optimizer: {e}")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "visualizations")

app = FastAPI(title="Process Optimization API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Legacy name-based API removed; only JSON payload endpoints are supported.


def write_temp_process_json(payload: Dict[str, Any]) -> Dict[str, str]:
    """Write provided process payload to a temporary JSON file.
    Returns dict with keys: path, id, name
    """
    # Ensure outputs dir exists for easier debugging of temp files
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Extract id/name or synthesize if missing
    proc_id = str(payload.get("id") or "")
    proc_name = str(payload.get("name") or "")
    if not proc_id:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        proc_id = f"payload_{ts}"
    if not proc_name:
        proc_name = proc_id

    # Write to a temp json file under outputs/visualizations for traceability
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", prefix=f"{proc_id}_", dir=OUTPUTS_DIR)
    tmp_path = tmp.name
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    finally:
        try:
            tmp.close()
        except Exception:
            pass
    return {"path": tmp_path, "id": proc_id, "name": proc_name}


def run_optimizer_and_collect(process_json_path: str, process_id: Optional[str] = None) -> Dict[str, str]:
    """Run the RL optimizer, suppressing GUI openings, then return paths to the two PNG outputs.
    Returns dict with keys: alloc_png_path, summary_png_path
    """
    # Ensure outputs dir exists
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Monkeypatch to suppress opening files
    orig_startfile = getattr(_os, "startfile", None)
    orig_web_open = _webbrowser.open
    try:
        setattr(_os, "startfile", lambda *a, **k: None)
    except Exception:
        pass
    _webbrowser.open = lambda *a, **k: False

    # Run optimizer (blocking)
    try:
        optimizer.main([process_json_path])
    finally:
        # Restore
        if orig_startfile is not None:
            try:
                setattr(_os, "startfile", orig_startfile)
            except Exception:
                pass
        _webbrowser.open = orig_web_open

    # Identify latest files for the process id
    if not process_id:
        # Derive from filename
        try:
            with open(process_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            process_id = data.get("id")
        except Exception:
            process_id = None

    if not process_id:
        raise HTTPException(status_code=500, detail="Could not determine process id from JSON")

    # Find latest allocation and summary PNGs
    alloc_pattern = os.path.join(OUTPUTS_DIR, f"{process_id}_alloc_charts_*.png")
    summary_pattern = os.path.join(OUTPUTS_DIR, f"{process_id}_summary_*.png")
    alloc_files = glob.glob(alloc_pattern)
    summary_files = glob.glob(summary_pattern)
    if not alloc_files or not summary_files:
        raise HTTPException(status_code=500, detail="Expected output PNGs not found after optimization")

    alloc_png = max(alloc_files, key=os.path.getmtime)
    summary_png = max(summary_files, key=os.path.getmtime)
    return {"alloc_png_path": alloc_png, "summary_png_path": summary_png}


def encode_png_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


@app.get("/health")
async def health():
    return {"status": "ok"}


# Legacy name-based endpoint /optimize removed.


@app.post("/optimize/payload")
async def optimize_payload(payload: Dict[str, Any]):
    """Accept full process JSON payload, write to temp file, optimize, return base64 charts."""
    if not isinstance(payload, dict) or not payload:
        raise HTTPException(status_code=400, detail="Request body must be a non-empty JSON object")

    meta = await asyncio.to_thread(write_temp_process_json, payload)
    paths = await asyncio.to_thread(run_optimizer_and_collect, meta["path"], meta["id"])

    alloc_b64 = await asyncio.to_thread(encode_png_base64, paths["alloc_png_path"])
    summary_b64 = await asyncio.to_thread(encode_png_base64, paths["summary_png_path"])

    return {
        "process_id": meta["id"],
        "process_name": meta["name"],
        "alloc_chart": {
            "filename": os.path.basename(paths["alloc_png_path"]),
            "path": os.path.abspath(paths["alloc_png_path"]),
            "base64": alloc_b64,
        },
        "summary_chart": {
            "filename": os.path.basename(paths["summary_png_path"]),
            "path": os.path.abspath(paths["summary_png_path"]),
            "base64": summary_b64,
        },
    }


@app.post("/optimize/payload/alloc_png")
async def optimize_payload_alloc_png(payload: Dict[str, Any]):
    """Accept full process JSON payload and return allocation chart as PNG file."""
    if not isinstance(payload, dict) or not payload:
        raise HTTPException(status_code=400, detail="Request body must be a non-empty JSON object")

    meta = await asyncio.to_thread(write_temp_process_json, payload)
    paths = await asyncio.to_thread(run_optimizer_and_collect, meta["path"], meta["id"])
    alloc_path = paths["alloc_png_path"]
    return FileResponse(
        alloc_path,
        media_type="image/png",
        filename=os.path.basename(alloc_path),
    )


@app.post("/optimize/payload/summary_png")
async def optimize_payload_summary_png(payload: Dict[str, Any]):
    """Accept full process JSON payload and return summary chart as PNG file."""
    if not isinstance(payload, dict) or not payload:
        raise HTTPException(status_code=400, detail="Request body must be a non-empty JSON object")

    meta = await asyncio.to_thread(write_temp_process_json, payload)
    paths = await asyncio.to_thread(run_optimizer_and_collect, meta["path"], meta["id"])
    summary_path = paths["summary_png_path"]
    return FileResponse(
        summary_path,
        media_type="image/png",
        filename=os.path.basename(summary_path),
    )


# Legacy name-based endpoints (/optimize/alloc_png, /optimize/summary_png, /optimize/bundle) removed.


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("API.main:app", host="0.0.0.0", port=8000, reload=True)
