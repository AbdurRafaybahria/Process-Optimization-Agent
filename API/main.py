import os
import sys
import asyncio
import base64
import tempfile
import json
import glob
import logging
import traceback
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system environment variables

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure project root is on sys.path so we can import run_rl_optimizer
PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import CMS integration modules
from process_optimization_agent.cms_client import CMSClient
from process_optimization_agent.cms_transformer import CMSDataTransformer
from process_optimization_agent.intelligent_optimizer import IntelligentOptimizer

import webbrowser as _webbrowser
import os as _os

try:
    import run_rl_optimizer as optimizer
except Exception as e:
    raise RuntimeError(f"Failed to import run_rl_optimizer: {e}")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "visualization_outputs")

# Default CMS configuration
# Use deployed CMS URL if available, fallback to localhost
DEFAULT_CMS_URL = os.getenv("REACT_APP_BASE_URL", "https://fyp-cms-frontend.vercel.app")

app = FastAPI(title="Process Optimization API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fyp-cms-frontend.vercel.app",
        "http://localhost:3000",
        "https://crystalsystemcms-testing-e377.up.railway.app",
        "https://crystalsystemcms-production.up.railway.app",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Only CMS process ID endpoints are supported. Payload endpoints have been removed.


def detect_data_format(payload: Dict[str, Any]) -> str:
    """
    Detect if the payload is in CMS format or Agent format.
    Returns 'cms' or 'agent'
    """
    # Agent format has 'tasks' and 'resources' at root level
    # CMS format has 'process_task' and nested structure
    if 'tasks' in payload and 'resources' in payload:
        return 'agent'
    elif 'process_task' in payload or 'company' in payload:
        return 'cms'
    else:
        # Default to agent format if unclear
        return 'agent'


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
    logger.info(f"Starting optimization for process file: {process_json_path}")
    
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

    # Run optimizer (blocking) with timeout protection
    try:
        logger.info("Running test_process_detection.py...")
        logger.info(f"Process JSON path: {process_json_path}")
        logger.info(f"Process ID: {process_id}")
        
        # Ensure output directory exists
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        logger.info(f"Output directory ensured: {OUTPUTS_DIR}")
        
        # Run the test_process_detection.py as a subprocess with timeout
        cmd = [sys.executable, "test_process_detection.py", process_json_path]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            logger.error(f"test_process_detection.py failed with return code {result.returncode}")
            logger.error(f"STDOUT:\n{result.stdout}")
            logger.error(f"STDERR:\n{result.stderr}")
            raise HTTPException(
                status_code=500, 
                detail=f"Optimization failed. Return code: {result.returncode}. Error: {result.stderr[:500]}"
            )
        
        logger.info("test_process_detection.py completed successfully")
        logger.info(f"Optimizer STDOUT (last 500 chars): {result.stdout[-500:]}")
        if result.stderr:
            logger.warning(f"Optimizer STDERR: {result.stderr}")
        
    except subprocess.TimeoutExpired:
        logger.error("Optimizer timed out after 5 minutes")
        raise HTTPException(status_code=500, detail="Optimization timed out after 5 minutes")
    except Exception as e:
        logger.error(f"Optimizer failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
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

    # Find latest allocation and summary PNGs (healthcare, insurance, or manufacturing format)
    # test_process_detection.py generates files like:
    # healthcare_allocation_{process_id}.png, insurance_allocation_{process_id}.png, or manufacturing_allocation_{process_id}.png
    alloc_patterns = [
        os.path.join(OUTPUTS_DIR, f"healthcare_allocation_{process_id}.png"),
        os.path.join(OUTPUTS_DIR, f"insurance_allocation_{process_id}.png"),
        os.path.join(OUTPUTS_DIR, f"manufacturing_allocation_{process_id}.png")
    ]
    summary_patterns = [
        os.path.join(OUTPUTS_DIR, f"healthcare_summary_{process_id}.png"),
        os.path.join(OUTPUTS_DIR, f"insurance_summary_{process_id}.png"),
        os.path.join(OUTPUTS_DIR, f"manufacturing_summary_{process_id}.png")
    ]
    
    # Find the most recent files
    alloc_files = []
    for pattern in alloc_patterns:
        if os.path.exists(pattern):
            alloc_files.append(pattern)
    
    summary_files = []
    for pattern in summary_patterns:
        if os.path.exists(pattern):
            summary_files.append(pattern)
    
    if not alloc_files or not summary_files:
        # List all files in the output directory for debugging
        logger.error(f"PNG files not found for process ID: {process_id}")
        logger.error(f"Output directory: {OUTPUTS_DIR}")
        if os.path.exists(OUTPUTS_DIR):
            all_files = os.listdir(OUTPUTS_DIR)
            logger.error(f"Available files: {all_files}")
            png_files = [f for f in all_files if f.endswith('.png')]
            logger.error(f"PNG files in directory: {png_files}")
        else:
            logger.error(f"Output directory does not exist: {OUTPUTS_DIR}")
        
        raise HTTPException(
            status_code=500, 
            detail=f"Expected output PNGs not found after optimization. Process ID: {process_id}. Check logs for details."
        )

    alloc_png = max(alloc_files, key=os.path.getmtime)
    summary_png = max(summary_files, key=os.path.getmtime)
    logger.info(f"Found allocation PNG: {alloc_png}")
    logger.info(f"Found summary PNG: {summary_png}")
    return {"alloc_png_path": alloc_png, "summary_png_path": summary_png}


def encode_png_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/cms/process/{process_id}")
async def get_cms_process(process_id: int, authorization: Optional[str] = Header(None)):
    """Fetch process data from CMS and return in agent-compatible format."""
    # Use provided token or authenticate dynamically
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
    
    # CMSClient will authenticate automatically if token is None
    client = CMSClient(base_url=DEFAULT_CMS_URL, bearer_token=token)
    transformer = CMSDataTransformer()
    
    # Fetch process from CMS
    cms_data = await asyncio.to_thread(client.get_process_with_relations, process_id)
    if not cms_data:
        raise HTTPException(status_code=404, detail=f"Process {process_id} not found in CMS")
    
    # Transform to agent format
    agent_format = transformer.transform_process(cms_data)
    return agent_format


@app.post("/cms/optimize/{process_id}")
async def optimize_cms_process(process_id: int, authorization: Optional[str] = Header(None)):
    """Fetch process from CMS, optimize it, and return results."""
    # Use provided token or authenticate dynamically
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
    
    # CMSClient will authenticate automatically if token is None
    client = CMSClient(base_url=DEFAULT_CMS_URL, bearer_token=token)
    transformer = CMSDataTransformer()
    
    # Fetch process from CMS
    cms_data = await asyncio.to_thread(client.get_process_with_relations, process_id)
    if not cms_data:
        raise HTTPException(status_code=404, detail=f"Process {process_id} not found in CMS")
    
    # Transform to agent format
    agent_format = transformer.transform_process(cms_data)
    
    # Write to temp file and optimize
    meta = await asyncio.to_thread(write_temp_process_json, agent_format)
    paths = await asyncio.to_thread(run_optimizer_and_collect, meta["path"], meta["id"])
    
    alloc_b64 = await asyncio.to_thread(encode_png_base64, paths["alloc_png_path"])
    summary_b64 = await asyncio.to_thread(encode_png_base64, paths["summary_png_path"])
    
    return {
        "process_id": str(process_id),
        "process_name": agent_format.get("process_name", ""),
        "company": agent_format.get("company", ""),
        "original_cms_data": cms_data,
        "transformed_data": agent_format,
        "optimization_results": {
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
    }



@app.post("/cms/optimize/{process_id}/alloc_png")
async def optimize_cms_process_alloc_png(process_id: int, authorization: Optional[str] = Header(None)):
    """Fetch process from CMS, optimize it, and return allocation chart as PNG file."""
    # Use provided token or authenticate dynamically
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
    
    # CMSClient will authenticate automatically if token is None
    client = CMSClient(base_url=DEFAULT_CMS_URL, bearer_token=token)
    transformer = CMSDataTransformer()
    
    # Fetch process from CMS
    cms_data = await asyncio.to_thread(client.get_process_with_relations, process_id)
    if not cms_data:
        raise HTTPException(status_code=404, detail=f"Process {process_id} not found in CMS")
    
    # Transform to agent format
    agent_format = transformer.transform_process(cms_data)
    
    # Write to temp file and optimize
    meta = await asyncio.to_thread(write_temp_process_json, agent_format)
    paths = await asyncio.to_thread(run_optimizer_and_collect, meta["path"], meta["id"])
    
    alloc_path = paths["alloc_png_path"]
    return FileResponse(
        alloc_path,
        media_type="image/png",
        filename=f"process_{process_id}_allocation_chart.png",
    )


@app.post("/cms/optimize/{process_id}/summary_png")
async def optimize_cms_process_summary_png(process_id: int, authorization: Optional[str] = Header(None)):
    """Fetch process from CMS, optimize it, and return summary chart as PNG file."""
    # Use provided token or authenticate dynamically
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
    
    # CMSClient will authenticate automatically if token is None
    client = CMSClient(base_url=DEFAULT_CMS_URL, bearer_token=token)
    transformer = CMSDataTransformer()
    
    # Fetch process from CMS
    cms_data = await asyncio.to_thread(client.get_process_with_relations, process_id)
    if not cms_data:
        raise HTTPException(status_code=404, detail=f"Process {process_id} not found in CMS")
    
    # Transform to agent format
    agent_format = transformer.transform_process(cms_data)
    
    # Write to temp file and optimize
    meta = await asyncio.to_thread(write_temp_process_json, agent_format)
    paths = await asyncio.to_thread(run_optimizer_and_collect, meta["path"], meta["id"])
    
    summary_path = paths["summary_png_path"]
    return FileResponse(
        summary_path,
        media_type="image/png",
        filename=f"process_{process_id}_summary_chart.png",
    )


# Payload alloc_png endpoint removed


# Payload summary_png endpoint removed


# Non-CMS endpoints removed - all optimization must go through CMS-aligned endpoints


# Legacy name-based endpoints (/optimize/alloc_png, /optimize/summary_png, /optimize/bundle) removed.


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("API.main:app", host="0.0.0.0", port=8000, reload=True)
