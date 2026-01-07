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
from process_optimization_agent import CMSClient, CMSDataTransformer, ProcessValidationError, IntelligentOptimizer
from process_optimization_agent.Optimization.multi_job_resolver import MultiJobResolver, resolve_multi_job_tasks, CostOptimizer, optimize_process_cost

import webbrowser as _webbrowser
import os as _os

try:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
    import run_rl_optimizer as optimizer
except Exception as e:
    logger.warning(f"Failed to import run_rl_optimizer: {e}. Some features may be unavailable.")
    optimizer = None

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "visualization_outputs")

# Default CMS configuration
# Use deployed CMS URL if available, fallback to production server
# Note: This should point to the BACKEND API, not the frontend
DEFAULT_CMS_URL = os.getenv("REACT_APP_BASE_URL", "https://server-digitaltwin-enterprise-production.up.railway.app")

app = FastAPI(title="Process Optimization API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fyp-cms-frontend.vercel.app",
        "http://localhost:3000",
        "https://crystalsystemcms-testing-e377.up.railway.app",
        "https://crystalsystemcms-production.up.railway.app",
        "https://crystalsystemcms.up.railway.app",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Only CMS process ID endpoints are supported. Payload endpoints have been removed.


def _build_improvements_section(
    time_saved_percentage: float,
    current_total_cost: float,
    optimized_total_cost: float,
    job_resolution_cost_summary: Optional[Dict[str, Any]],
    multi_job_resolutions: list,
    cost_optimization_result: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build a comprehensive improvements section that includes:
    - Time efficiency from parallel processing
    - Cost savings from job resolution (removing redundant jobs)
    - Cost savings from job replacement (finding cheaper alternatives)
    - Detailed breakdown of optimizations per task
    """
    
    # Time improvement
    time_improvement = {
        "description": f"{time_saved_percentage:.1f}% faster through parallel task execution",
        "method": "Running independent tasks simultaneously instead of sequentially"
    }
    
    # Cost improvements - Job Resolution (removing redundant parallel jobs)
    job_resolution_savings = 0.0
    job_resolution_details = []
    
    if job_resolution_cost_summary:
        job_resolution_savings = job_resolution_cost_summary.get('total_savings', 0)
        savings_pct = job_resolution_cost_summary.get('savings_percentage', 0)
        
        # Build details from resolutions
        for resolution in multi_job_resolutions:
            cost_analysis = resolution.get('cost_analysis', {})
            task_savings = cost_analysis.get('savings', 0)
            
            if task_savings > 0:
                detail = {
                    "task_id": str(resolution.get('task_id', '')),
                    "task_name": resolution.get('task_name', ''),
                    "optimization_type": resolution.get('resolution', ''),
                    "savings": round(task_savings, 2),
                    "explanation": ""
                }
                
                if resolution.get('resolution') == 'best_fit':
                    # Best fit - removed redundant jobs
                    kept = resolution.get('kept_jobs', [])
                    removed = resolution.get('removed_jobs', [])
                    kept_names = [j.get('name', '') for j in kept]
                    removed_names = [f"{j.get('name', '')} (${j.get('hourlyRate', 0)}/hr)" for j in removed]
                    detail["explanation"] = f"Kept '{', '.join(kept_names)}' as best fit. Removed redundant jobs: {', '.join(removed_names)}"
                    detail["kept_job"] = kept[0].get('name', '') if kept else ''
                    detail["removed_jobs"] = [j.get('name', '') for j in removed]
                    
                elif resolution.get('resolution') == 'split':
                    # Split - task divided into sub-tasks with specific jobs
                    sub_tasks = resolution.get('sub_tasks', [])
                    removed = resolution.get('removed_jobs', [])
                    sub_task_info = [f"{st.get('name', '')} → {st.get('job', '')} (${st.get('hourlyRate', 0)}/hr)" for st in sub_tasks]
                    removed_names = [f"{j.get('name', '')} (${j.get('hourlyRate', 0)}/hr)" for j in removed if j.get('name') not in [st.get('job') for st in sub_tasks]]
                    detail["explanation"] = f"Task split into specialized sub-tasks: {'; '.join(sub_task_info)}. Removed: {', '.join(removed_names) if removed_names else 'None'}"
                    detail["sub_tasks"] = [{"name": st.get('name'), "assigned_to": st.get('job')} for st in sub_tasks]
                
                job_resolution_details.append(detail)
    
    # Cost improvements - Job Replacement (finding cheaper alternatives)
    job_replacement_savings = 0.0
    job_replacement_details = []
    
    if cost_optimization_result:
        job_replacement_savings = cost_optimization_result.get('total_savings', 0)
        replacements = cost_optimization_result.get('replacements', [])
        
        for replacement in replacements:
            original_job = replacement.get('original_job', {})
            new_job = replacement.get('new_job', {})
            task_savings = replacement.get('cost_savings', 0)
            
            detail = {
                "task_id": str(replacement.get('task_id', '')),
                "task_name": replacement.get('task_name', ''),
                "optimization_type": "job_replacement",
                "savings": round(task_savings, 2),
                "original_job": original_job.get('name', ''),
                "original_rate": original_job.get('hourlyRate', 0),
                "new_job": new_job.get('name', ''),
                "new_rate": new_job.get('hourlyRate', 0),
                "explanation": f"Replaced '{original_job.get('name', '')}' (${original_job.get('hourlyRate', 0)}/hr) with cheaper qualified job '{new_job.get('name', '')}' (${new_job.get('hourlyRate', 0)}/hr)"
            }
            job_replacement_details.append(detail)
    
    # Total cost savings
    total_cost_savings = job_resolution_savings + job_replacement_savings
    
    # Build the improvements object
    improvements = {
        "time_efficiency": {
            "improvement": f"{time_saved_percentage:.1f}% faster",
            "method": "Parallel task execution",
            "description": "Running independent tasks simultaneously instead of sequentially reduces total process time"
        },
        "cost_efficiency": {
            "total_savings": round(total_cost_savings, 2),
            "total_savings_formatted": f"${total_cost_savings:.2f}",
            "breakdown": {
                "job_resolution_savings": {
                    "amount": round(job_resolution_savings, 2),
                    "amount_formatted": f"${job_resolution_savings:.2f}",
                    "percentage": round(job_resolution_cost_summary.get('savings_percentage', 0), 2) if job_resolution_cost_summary else 0,
                    "method": "Removing redundant parallel job assignments",
                    "description": "When multiple jobs were assigned to a single task (working in parallel), we kept only the best-fit job(s) and removed redundant ones",
                    "tasks_optimized": len(job_resolution_details),
                    "details": job_resolution_details
                },
                "job_replacement_savings": {
                    "amount": round(job_replacement_savings, 2),
                    "amount_formatted": f"${job_replacement_savings:.2f}",
                    "percentage": round(cost_optimization_result.get('savings_percentage', 0), 2) if cost_optimization_result else 0,
                    "method": "Replacing jobs with cheaper qualified alternatives",
                    "description": "Finding jobs with lower hourly rates that still meet the required skill threshold (≥90% match)",
                    "tasks_optimized": len(job_replacement_details),
                    "details": job_replacement_details
                }
            }
        },
        "resource_utilization": {
            "improvement": "Improved",
            "method": "Parallel execution and skill-based assignment",
            "description": "Resources are utilized more efficiently through parallel task execution and matching tasks to specialists with the best skill fit"
        },
        "process_flexibility": {
            "improvement": "Enhanced",
            "method": "1:1 task-job relationship",
            "description": "Clear accountability with each task assigned to exactly one qualified job, making the process easier to manage and scale"
        },
        "summary": {
            "time_saved": f"{time_saved_percentage:.1f}%",
            "cost_saved": f"${total_cost_savings:.2f}",
            "optimization_methods": []
        }
    }
    
    # Build summary of optimization methods used
    if time_saved_percentage > 0:
        improvements["summary"]["optimization_methods"].append("Parallel task execution")
    if job_resolution_savings > 0:
        improvements["summary"]["optimization_methods"].append(f"Job resolution: removed redundant jobs (${job_resolution_savings:.2f} saved)")
    if job_replacement_savings > 0:
        improvements["summary"]["optimization_methods"].append(f"Job replacement: found cheaper alternatives (${job_replacement_savings:.2f} saved)")
    if not improvements["summary"]["optimization_methods"]:
        improvements["summary"]["optimization_methods"].append("Process already optimized - no further improvements found")
    
    return improvements


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
        test_script = os.path.join(PROJECT_ROOT, "tests", "test_process_detection.py")
        cmd = [sys.executable, test_script, process_json_path]
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
    
    # Transform to agent format with validation
    try:
        agent_format = transformer.transform_process(cms_data)
    except ProcessValidationError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": e.error_code,
                "message": e.message,
                "process_id": process_id
            }
        )
    
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
    
    # Transform to agent format with validation
    try:
        agent_format = transformer.transform_process(cms_data)
    except ProcessValidationError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": e.error_code,
                "message": e.message,
                "process_id": process_id
            }
        )
    
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
    
    # Transform to agent format with validation
    try:
        agent_format = transformer.transform_process(cms_data)
    except ProcessValidationError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": e.error_code,
                "message": e.message,
                "process_id": process_id
            }
        )
    
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


@app.post("/cms/optimize/{process_id}/json")
async def optimize_cms_process_json(process_id: int, authorization: Optional[str] = Header(None)):
    """
    Fetch process from CMS, optimize it, and return complete optimization results in JSON format.
    
    Input: process_id
    Output: Complete optimization data including:
        - Process information
        - Current state (before optimization)
        - Optimized state (after optimization)  
        - Suggestions and recommendations
        - Task assignments
        - Parallel execution opportunities
        - Constraints and risks
        - Implementation steps
    """
    # Use provided token or authenticate dynamically
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
    
    # CMSClient will authenticate automatically if token is None
    client = CMSClient(base_url=DEFAULT_CMS_URL, bearer_token=token)
    transformer = CMSDataTransformer()
    
    try:
        # Fetch process from CMS
        cms_data = await asyncio.to_thread(client.get_process_with_relations, process_id)
        if not cms_data:
            raise HTTPException(status_code=404, detail=f"Process {process_id} not found in CMS")
        
        # CALCULATE ORIGINAL COST BEFORE JOB RESOLUTION
        # We'll calculate this after job resolution by adding back the savings
        # This is more accurate than trying to parse raw CMS data
        original_total_cost = 0.0  # Will be calculated after we know the resolution savings
        
        # Fetch jobs with their real skills from CMS
        # This provides actual job skills for better job resolution
        # Use get_jobs_for_process to filter only jobs assigned to this process
        jobs_with_skills = {}
        all_jobs_map = {}  # ALL jobs from CMS for cost optimization
        try:
            jobs_with_skills = await asyncio.to_thread(client.get_jobs_for_process, cms_data)
            if jobs_with_skills:
                print(f"[INFO] Fetched {len(jobs_with_skills)} jobs with skills for process {process_id}")
                # Debug: Print skills found for each job
                for job_id, job_data in jobs_with_skills.items():
                    skills = job_data.get('skills', [])
                    skill_names = [s.get('name') for s in skills]
                    print(f"[DEBUG] Job {job_data.get('name')} (ID:{job_id}) has skills: {skill_names}")
            
            # Fetch ALL jobs from CMS for cost optimization comparison
            all_jobs_map = await asyncio.to_thread(client.get_all_jobs_map_with_skills)
            print(f"[INFO] Fetched {len(all_jobs_map)} total jobs from CMS for cost optimization")
        except Exception as e:
            print(f"[WARNING] Could not fetch jobs with skills: {e}")
        
        # Resolve multi-job tasks BEFORE transformation
        # This ensures 1:1 task-job relationship before optimization
        # Pass real job skills data for accurate job matching
        resolver = MultiJobResolver(best_fit_threshold=0.90, jobs_with_skills=jobs_with_skills)
        resolved_cms_data = resolver.resolve_process(cms_data)
        
        # Extract resolution details for response
        multi_job_resolutions = resolved_cms_data.pop('_multi_job_resolutions', [])
        job_resolution_cost_summary = resolved_cms_data.pop('_job_resolution_cost_summary', None)
        
        # NOW calculate original cost using job resolution savings
        # Original cost = cost after resolution + resolution savings
        # This is more accurate than trying to parse raw CMS data
        if job_resolution_cost_summary and 'total_savings' in job_resolution_cost_summary:
            job_resolution_savings = job_resolution_cost_summary['total_savings']
            # We'll add this to the optimized cost later when we have it
            print(f"[COST] Job resolution savings: ${job_resolution_savings:.2f}")
        else:
            job_resolution_savings = 0.0
        
        # COST OPTIMIZATION: Find cheaper qualified jobs
        cost_optimization_result = None
        cost_optimization_savings = 0.0
        if all_jobs_map:
            try:
                resolved_cms_data, cost_opt_result = optimize_process_cost(
                    resolved_cms_data, 
                    all_jobs_map, 
                    skill_match_threshold=0.90
                )
                cost_optimization_result = resolved_cms_data.pop('_cost_optimization', None)
                cost_optimization_savings = cost_opt_result.total_savings if cost_opt_result else 0.0
                print(f"[INFO] Cost optimization complete: ${cost_optimization_savings:.2f} savings")
            except Exception as e:
                print(f"[WARNING] Cost optimization failed: {e}")
                cost_optimization_result = None
                cost_optimization_savings = 0.0
        
        # Transform to agent format with validation (using resolved data)
        try:
            agent_format = transformer.transform_process(resolved_cms_data)
        except ProcessValidationError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": e.error_code,
                    "message": e.message,
                    "process_id": process_id
                }
            )
        
        # Run optimization using IntelligentOptimizer
        from process_optimization_agent import ProcessIntelligence, ProcessType
        from process_optimization_agent.Optimization.models import Process, Task, Resource, Skill, SkillLevel
        
        # Convert agent_format to Process object
        process = Process(
            id=agent_format.get("id", str(process_id)),
            name=agent_format.get("process_name", ""),
            description=agent_format.get("description", ""),
            tasks=[],
            resources=[]
        )
        
        # Add tasks
        for task_data in agent_format.get("tasks", []):
            # Handle required_skills which can be either strings or dicts
            required_skills = []
            for skill in task_data.get("required_skills", []):
                if isinstance(skill, dict):
                    skill_name = skill.get("name", skill.get("skill_name", ""))
                else:
                    skill_name = str(skill)
                if skill_name:
                    required_skills.append(Skill(name=skill_name, level=SkillLevel.INTERMEDIATE))
            
            task = Task(
                id=task_data["id"],
                name=task_data["name"],
                description=task_data.get("description", ""),
                duration_hours=task_data["duration_hours"],
                required_skills=required_skills
            )
            process.tasks.append(task)
        
        # Calculate total required hours per resource skill to set adequate capacity
        total_hours_by_skill = {}
        for task in process.tasks:
            for skill in task.required_skills:
                skill_key = skill.name.lower().strip()
                total_hours_by_skill[skill_key] = total_hours_by_skill.get(skill_key, 0) + task.duration_hours
        
        # Add resources with adequate capacity
        for resource_data in agent_format.get("resources", []):
            # Handle skills which can be either strings or dicts
            skills = []
            for skill in resource_data.get("skills", []):
                if isinstance(skill, dict):
                    skill_name = skill.get("name", skill.get("skill_name", ""))
                else:
                    skill_name = str(skill)
                if skill_name:
                    skills.append(Skill(name=skill_name, level=SkillLevel.INTERMEDIATE))
            
            # Calculate required capacity for this resource based on its skills
            max_required_hours = 160.0  # Default minimum
            for skill in skills:
                skill_key = skill.name.lower().strip()
                if skill_key in total_hours_by_skill:
                    max_required_hours = max(max_required_hours, total_hours_by_skill[skill_key] * 1.2)  # 20% buffer
            
            resource = Resource(
                id=resource_data["id"],
                name=resource_data["name"],
                hourly_rate=resource_data.get("hourly_rate", 0),
                skills=skills,
                total_available_hours=max_required_hours  # Set adequate capacity
            )
            process.resources.append(resource)
        
        # Run intelligent optimization
        intelligent_optimizer = IntelligentOptimizer()
        optimization_result = await asyncio.to_thread(intelligent_optimizer.optimize, process)
        
        # Calculate metrics
        schedule = optimization_result.schedule
        
        # Calculate current state (sequential execution)
        # Original cost will be calculated after we have optimized cost
        current_total_time = sum(task.duration_hours for task in process.tasks)
        
        # Calculate optimized state from actual task assignments
        # This reflects cost AFTER job resolution and cost optimization
        if schedule and schedule.entries:
            optimized_total_time = max(entry.end_hour for entry in schedule.entries)
            # Calculate from actual assignments (already includes job resolution savings)
            optimized_total_cost = sum(
                (entry.end_hour - entry.start_hour) * 
                next((r.hourly_rate for r in process.resources if r.id == entry.resource_id), 50)
                for entry in schedule.entries
            )
        else:
            optimized_total_time = current_total_time
            optimized_total_cost = 0.0
        
        # NOW calculate original cost: optimized cost + ALL savings
        # This represents the TRUE original cost before any optimizations
        # Must include BOTH job resolution savings AND cost optimization savings
        total_savings = job_resolution_savings + cost_optimization_savings
        original_total_cost = optimized_total_cost + total_savings
        current_total_cost = original_total_cost
        
        print(f"[COST] Job resolution savings: ${job_resolution_savings:.2f}")
        print(f"[COST] Cost optimization savings: ${cost_optimization_savings:.2f}")
        print(f"[COST] Original cost BEFORE all optimizations: ${original_total_cost:.2f}")
        print(f"[COST] Optimized cost AFTER all optimizations: ${optimized_total_cost:.2f}")
        print(f"[COST] Total savings: ${total_savings:.2f}")
        
        # Calculate time saved
        time_saved = current_total_time - optimized_total_time
        time_saved_percentage = (time_saved / current_total_time * 100) if current_total_time > 0 else 0
        
        # Build task assignments
        task_assignments = []
        for entry in schedule.entries if schedule else []:
            task = next((t for t in process.tasks if t.id == entry.task_id), None)
            resource = next((r for r in process.resources if r.id == entry.resource_id), None)
            if task and resource:
                task_assignments.append({
                    "task_name": task.name,
                    "task_id": task.id,
                    "resource_name": resource.name,
                    "resource_id": resource.id,
                    "hourly_rate": resource.hourly_rate,
                    "duration_hours": entry.end_hour - entry.start_hour,
                    "duration_minutes": (entry.end_hour - entry.start_hour) * 60,
                    "start_time": entry.start_hour,
                    "end_time": entry.end_hour,
                    "cost": (entry.end_hour - entry.start_hour) * resource.hourly_rate
                })
        
        # Analyze execution patterns: parallel vs sequential
        parallel_tasks = []
        sequential_tasks = []
        parallel_execution_steps = []
        execution_pattern_analysis = {
            "total_tasks": len(process.tasks),
            "parallel_tasks_count": 0,
            "sequential_tasks_count": 0,
            "execution_mode": "unknown"
        }
        
        if schedule and schedule.entries:
            # Sort entries by end time to track completion order
            sorted_by_end = sorted(schedule.entries, key=lambda e: e.end_hour)
            
            # Create timeline events (start and end)
            timeline_events = []
            for entry in schedule.entries:
                task = next((t for t in process.tasks if t.id == entry.task_id), None)
                resource = next((r for r in process.resources if r.id == entry.resource_id), None)
                if task and resource:
                    timeline_events.append({
                        'time': entry.start_hour,
                        'type': 'start',
                        'task_id': entry.task_id,
                        'task_name': task.name,
                        'resource_name': resource.name,
                        'duration': entry.end_hour - entry.start_hour,
                        'end_time': entry.end_hour
                    })
                    timeline_events.append({
                        'time': entry.end_hour,
                        'type': 'end',
                        'task_id': entry.task_id,
                        'task_name': task.name,
                        'resource_name': resource.name
                    })
            
            # Sort by time, with ends before starts at the same time
            timeline_events.sort(key=lambda e: (e['time'], 0 if e['type'] == 'end' else 1))
            
            # Track active tasks at each step
            active_tasks = {}  # task_id -> task info
            step_number = 0
            previous_time = None
            
            for event in timeline_events:
                current_time = event['time']
                
                # Process the event
                if event['type'] == 'start':
                    active_tasks[event['task_id']] = {
                        'task_name': event['task_name'],
                        'resource_name': event['resource_name'],
                        'duration_hours': event['duration'],
                        'start_time': current_time,
                        'end_time': event['end_time']
                    }
                else:  # end event
                    if event['task_id'] in active_tasks:
                        del active_tasks[event['task_id']]
                
                # Create a step when the active task set changes
                # Skip if this is the same time as previous (batch changes together)
                if current_time != previous_time and active_tasks:
                    step_number += 1
                    
                    # Sort tasks by end time (tasks finishing sooner appear first)
                    sorted_active = sorted(
                        active_tasks.items(),
                        key=lambda x: x[1]['end_time']
                    )
                    
                    parallel_execution_steps.append({
                        "step": step_number,
                        "time": current_time,
                        "active_task_count": len(active_tasks),
                        "active_tasks": [
                            {
                                "task_name": info['task_name'],
                                "resource_name": info['resource_name'],
                                "duration_hours": info['duration_hours'],
                                "start_time": info['start_time'],
                                "end_time": info['end_time'],
                                "remaining_time": max(0, info['end_time'] - current_time)
                            }
                            for task_id, info in sorted_active
                        ],
                        "description": f"Step {step_number}: {len(active_tasks)} task(s) running in parallel at t={current_time:.2f}h"
                    })
                    
                    previous_time = current_time
            
            # Build network diagram structure for frontend
            network_diagram = {
                "start": {
                    "type": "start",
                    "label": "Start",
                    "time": 0.0
                },
                "stages": [],
                "finish": {
                    "type": "finish",
                    "label": "Finish",
                    "time": 0.0
                }
            }
            
            # Track all unique time points where tasks start or end
            time_points = set()
            for entry in schedule.entries:
                time_points.add(entry.start_hour)
                time_points.add(entry.end_hour)
            
            # Sort time points to create stages
            sorted_times = sorted(time_points)
            
            # Build stages based on active tasks at each time point
            stage_number = 0
            for i, current_time in enumerate(sorted_times):
                # Find all tasks active at this time
                active_at_time = []
                for entry in schedule.entries:
                    if entry.start_hour <= current_time < entry.end_hour:
                        task = next((t for t in process.tasks if t.id == entry.task_id), None)
                        resource = next((r for r in process.resources if r.id == entry.resource_id), None)
                        if task and resource:
                            active_at_time.append({
                                "task_id": task.id,
                                "task_name": task.name,
                                "resource_name": resource.name,
                                "duration_hours": entry.end_hour - entry.start_hour,
                                "start_time": entry.start_hour,
                                "end_time": entry.end_hour,
                                "progress": ((current_time - entry.start_hour) / (entry.end_hour - entry.start_hour) * 100) if entry.end_hour > entry.start_hour else 100,
                                "has_dependencies": bool(task.dependencies) if task.dependencies else False
                            })
                
                # Only create a stage if there are active tasks
                if active_at_time:
                    stage_number += 1
                    
                    # Determine execution type for this stage
                    execution_type = "parallel" if len(active_at_time) > 1 else "sequential"
                    
                    network_diagram["stages"].append({
                        "stage": stage_number,
                        "time": current_time,
                        "execution_type": execution_type,
                        "active_task_count": len(active_at_time),
                        "tasks": active_at_time,
                        "description": f"Stage {stage_number}: {len(active_at_time)} task(s) active at t={current_time:.2f}h"
                    })
            
            # Set finish time
            if schedule.entries:
                network_diagram["finish"]["time"] = max(entry.end_hour for entry in schedule.entries)
            
            # Group tasks by start time for execution pattern analysis
            time_groups = {}
            for entry in schedule.entries:
                if entry.start_hour not in time_groups:
                    time_groups[entry.start_hour] = []
                time_groups[entry.start_hour].append(entry)
            
            # Identify parallel vs sequential execution patterns
            # First, analyze the actual schedule to see what's happening
            parallel_task_ids = set()
            sequential_task_ids = set()
            
            # Build a map of when each task actually starts
            task_start_times = {}
            for entry in schedule.entries:
                task_start_times[entry.task_id] = entry.start_hour
            
            # Analyze each time group
            for start_time, entries in sorted(time_groups.items()):
                task_info_list = []
                for entry in entries:
                    task = next((t for t in process.tasks if t.id == entry.task_id), None)
                    resource = next((r for r in process.resources if r.id == entry.resource_id), None)
                    if task and resource:
                        # Check if dependencies were actually respected
                        deps_respected = True
                        dependency_chain = []
                        if task.dependencies:
                            for dep_id in task.dependencies:
                                if dep_id in task_start_times:
                                    # Dependency should have started before this task
                                    dep_start = task_start_times[dep_id]
                                    if dep_start >= entry.start_hour:
                                        deps_respected = False
                                    dependency_chain.append(dep_id)
                        
                        task_info_list.append({
                            "task_id": task.id,
                            "task_name": task.name,
                            "resource_name": resource.name,
                            "duration_hours": entry.end_hour - entry.start_hour,
                            "start_time": entry.start_hour,
                            "end_time": entry.end_hour,
                            "has_dependencies": bool(task.dependencies),
                            "dependencies_respected": deps_respected,
                            "dependency_chain": dependency_chain
                        })
                
                if len(entries) > 1:
                    # Multiple tasks starting at same time
                    # Check if any have dependencies - if so, they're running in parallel incorrectly
                    has_deps = any(info["has_dependencies"] for info in task_info_list)
                    task_names = [info["task_name"] for info in task_info_list]
                    
                    parallel_tasks.append({
                        "start_time": start_time,
                        "task_count": len(entries),
                        "tasks": task_names,
                        "task_details": task_info_list,
                        "note": "Some tasks have dependencies but are running in parallel" if has_deps else "Independent tasks running in parallel"
                    })
                    for info in task_info_list:
                        parallel_task_ids.add(info["task_id"])
                else:
                    # Single task at this time point = sequential execution
                    info = task_info_list[0]
                    
                    # Determine the reason for sequential execution
                    reason = "No parallel opportunities at this time"
                    if info["has_dependencies"]:
                        if info["dependencies_respected"]:
                            dep_names = [next((t.name for t in process.tasks if t.id == dep_id), dep_id) 
                                        for dep_id in info["dependency_chain"]]
                            reason = f"Must wait for dependencies: {', '.join(dep_names)}"
                        else:
                            reason = "Has dependencies (may not be properly enforced in schedule)"
                    
                    sequential_tasks.append({
                        "task_id": info["task_id"],
                        "task_name": info["task_name"],
                        "resource_name": info["resource_name"],
                        "duration_hours": info["duration_hours"],
                        "start_time": info["start_time"],
                        "end_time": info["end_time"],
                        "has_dependencies": info["has_dependencies"],
                        "reason": reason
                    })
                    sequential_task_ids.add(info["task_id"])
            
            # Check if there are tasks with dependencies that weren't scheduled
            scheduled_task_ids = {entry.task_id for entry in schedule.entries}
            unscheduled_tasks = [t for t in process.tasks if t.id not in scheduled_task_ids]
            
            # Determine overall execution mode
            execution_pattern_analysis["parallel_tasks_count"] = len(parallel_task_ids)
            execution_pattern_analysis["sequential_tasks_count"] = len(sequential_task_ids)
            execution_pattern_analysis["unscheduled_tasks_count"] = len(unscheduled_tasks)
            
            if unscheduled_tasks:
                execution_pattern_analysis["unscheduled_tasks"] = [
                    {
                        "task_id": t.id,
                        "task_name": t.name,
                        "has_dependencies": bool(t.dependencies),
                        "reason": "Not scheduled - may have resource or dependency issues"
                    }
                    for t in unscheduled_tasks
                ]
            
            # Determine execution mode
            total_scheduled = len(scheduled_task_ids)
            if total_scheduled == 0:
                execution_pattern_analysis["execution_mode"] = "none"
                execution_pattern_analysis["description"] = "No tasks were successfully scheduled"
            elif len(parallel_task_ids) == total_scheduled and len(parallel_task_ids) > 0:
                execution_pattern_analysis["execution_mode"] = "fully_parallel"
                execution_pattern_analysis["description"] = "All scheduled tasks run in parallel"
            elif len(sequential_task_ids) == total_scheduled:
                execution_pattern_analysis["execution_mode"] = "fully_sequential"
                execution_pattern_analysis["description"] = "All tasks run sequentially (one at a time)"
            else:
                execution_pattern_analysis["execution_mode"] = "mixed"
                execution_pattern_analysis["description"] = f"{len(parallel_task_ids)} tasks run in parallel, {len(sequential_task_ids)} tasks run sequentially"
            
            # Add warning if dependencies exist but tasks are running in parallel
            if parallel_tasks:
                tasks_with_deps_in_parallel = sum(1 for group in parallel_tasks 
                                                  for task in group.get("task_details", []) 
                                                  if task.get("has_dependencies"))
                if tasks_with_deps_in_parallel > 0:
                    execution_pattern_analysis["warning"] = f"{tasks_with_deps_in_parallel} tasks with dependencies are running in parallel - dependencies may not be properly enforced"
        
        # Generate suggestions
        suggestions = []
        
        # Suggestion 1: Parallel Processing
        if time_saved > 0:
            suggestions.append({
                "id": "opt_001",
                "title": "Implement Parallel Processing",
                "description": f"Run {len(process.tasks)} independent tasks simultaneously instead of sequentially",
                "type": "parallel_processing",
                "category": "quick_win",
                "impact": {
                    "time_saved": f"{time_saved:.2f} hours",
                    "time_saved_percentage": f"{time_saved_percentage:.1f}%",
                    "cost_change": f"${optimized_total_cost - current_total_cost:.2f}",
                    "cost_impact": "Increased" if optimized_total_cost > current_total_cost else "Reduced"
                },
                "implementation": {
                    "difficulty_level": 2,
                    "difficulty_stars": "⭐⭐",
                    "risk_level": "LOW",
                    "estimated_time": "2-3 weeks"
                },
                "details": {
                    "what_changes": "Tasks will run simultaneously instead of sequentially",
                    "resources_needed": [
                        "2 hours manager time for coordination",
                        f"{len(process.resources)} hours training (1 hour per specialist)"
                    ],
                    "success_metrics": [
                        f"Process completion time < {optimized_total_time + 2} hours",
                        "No increase in error rates",
                        "All specialists can work independently"
                    ],
                    "risks": [
                        "Coordination challenges between specialists",
                        "Quality issues from lack of sequential verification"
                    ],
                    "mitigation": [
                        "Daily 15-minute sync meetings for first month",
                        "Quality checkpoints at 25%, 50%, 75% completion"
                    ]
                },
                "implementation_steps": [
                    {
                        "phase": 1,
                        "title": "Preparation",
                        "duration": "Week 1-2",
                        "tasks": [
                            f"Meet with all {len(process.resources)} specialists",
                            "Explain new parallel workflow",
                            "Set up communication protocols"
                        ]
                    },
                    {
                        "phase": 2,
                        "title": "Pilot Testing",
                        "duration": "Week 3-4",
                        "tasks": [
                            "Run pilot with 3-5 test processes",
                            "Monitor for issues and bottlenecks",
                            "Collect feedback from team"
                        ]
                    },
                    {
                        "phase": 3,
                        "title": "Full Implementation",
                        "duration": "Week 5-6",
                        "tasks": [
                            "Roll out to all processes",
                            "Intensive monitoring and support",
                            "Document lessons learned"
                        ]
                    }
                ],
                "status": "pending",
                "can_implement_independently": True
            })
        
        # Suggestion 2: Resource Optimization (if there are cost savings opportunities)
        if len(process.resources) > 1:
            suggestions.append({
                "id": "opt_002",
                "title": "Optimize Resource Skill Matching",
                "description": "Better match tasks to specialist skills for improved efficiency",
                "type": "resource_optimization",
                "category": "quick_win",
                "impact": {
                    "time_saved": "0 hours",
                    "cost_impact": "Potential cost reduction through better matching",
                    "quality_improvement": "Higher quality through specialized assignments"
                },
                "implementation": {
                    "difficulty_level": 1,
                    "difficulty_stars": "⭐",
                    "risk_level": "LOW",
                    "estimated_time": "1 week"
                },
                "status": "pending",
                "can_implement_independently": True
            })
        
        # Detect process type for patient journey
        detected_process_type = getattr(optimization_result, "process_type", None)
        is_healthcare = detected_process_type and detected_process_type.value.lower() == "healthcare"
        
        # Build patient journey data for healthcare processes
        patient_journey = None
        if is_healthcare and schedule and schedule.entries:
            # Extract patient-facing tasks (tasks with user involvement)
            from process_optimization_agent.Optimization.models import UserInvolvement
            
            patient_steps = []
            cumulative_time = 0.0
            cumulative_cost = 0.0
            
            # Sort entries by start time to show journey progression
            sorted_entries = sorted(schedule.entries, key=lambda e: e.start_hour)
            
            for entry in sorted_entries:
                task = next((t for t in process.tasks if t.id == entry.task_id), None)
                resource = next((r for r in process.resources if r.id == entry.resource_id), None)
                
                if task and resource:
                    # Include all tasks in patient journey (they're all part of the process)
                    duration = entry.end_hour - entry.start_hour
                    cost = duration * resource.hourly_rate
                    
                    # Determine waiting time (gap between previous task end and current task start)
                    waiting_time = entry.start_hour - cumulative_time if cumulative_time > 0 else 0
                    
                    cumulative_time = entry.end_hour
                    cumulative_cost += cost
                    
                    # Determine involvement type
                    involvement_type = getattr(task, 'user_involvement', UserInvolvement.DIRECT).value
                    
                    patient_steps.append({
                        "step_number": len(patient_steps) + 1,
                        "task_id": task.id,
                        "task_name": task.name,
                        "resource_id": resource.id,
                        "resource_name": resource.name,
                        "involvement_type": involvement_type,
                        "start_time": entry.start_hour,
                        "duration_hours": duration,
                        "duration_minutes": duration * 60,
                        "end_time": entry.end_hour,
                        "waiting_time_hours": waiting_time,
                        "waiting_time_minutes": waiting_time * 60,
                        "cumulative_time_hours": cumulative_time,
                        "cumulative_time_minutes": cumulative_time * 60,
                        "step_cost": cost,
                        "cumulative_cost": cumulative_cost
                    })
            
            # Calculate journey metrics
            total_active_time = sum(step["duration_hours"] for step in patient_steps)
            total_waiting_time = sum(step["waiting_time_hours"] for step in patient_steps)
            total_journey_time = cumulative_time
            
            patient_journey = {
                "enabled": True,
                "total_steps": len(patient_steps),
                "total_journey_time_hours": total_journey_time,
                "total_journey_time_minutes": total_journey_time * 60,
                "total_active_time_hours": total_active_time,
                "total_active_time_minutes": total_active_time * 60,
                "total_waiting_time_hours": total_waiting_time,
                "total_waiting_time_minutes": total_waiting_time * 60,
                "total_cost": cumulative_cost,
                "patient_satisfaction_score": 1.0 - min(1.0, total_waiting_time / total_journey_time) if total_journey_time > 0 else 1.0,
                "number_of_touchpoints": len(patient_steps),
                "number_of_resource_changes": len(set(step["resource_id"] for step in patient_steps)) - 1 if patient_steps else 0,
                "steps": patient_steps,
                "journey_description": f"Patient journey with {len(patient_steps)} steps over {total_journey_time:.2f} hours"
            }
        
        # Build the complete response
        response = {
            "process_id": str(process_id),
            "process_name": process.name,
            "company": agent_format.get("company", ""),
            # High-level process classification (which domain this process belongs to)
            "process_type": {
                # e.g. "manufacturing", "insurance", "healthcare", etc.
                "type": detected_process_type.value if detected_process_type else "GENERIC",
                # Overall confidence from the intelligent optimizer
                "confidence": getattr(optimization_result, "confidence", 0),
                # Optimization strategy actually used, e.g. "parallel_production", "insurance_workflow"
                "strategy": getattr(optimization_result, "optimization_strategy", None).value if getattr(optimization_result, "optimization_strategy", None) else "",
                # Scenario type for insurance workflows (e.g. "STANDARD_BILLING"), if available
                "scenario_type": (optimization_result.admin_metrics.get("scenario_type")
                                   if getattr(optimization_result, "admin_metrics", None)
                                   and isinstance(optimization_result.admin_metrics, dict)
                                   else None)
            },
            "optimization_summary": {
                "total_suggestions": len(suggestions),
                "potential_time_saved": f"{time_saved:.2f} hours",
                "potential_time_saved_percentage": f"{time_saved_percentage:.1f}%",
                "cost_impact": "Increased" if optimized_total_cost > current_total_cost else "Reduced",
                "implementation_complexity": "Medium",
                "quick_wins_count": sum(1 for s in suggestions if s.get("category") == "quick_win"),
                "long_term_count": sum(1 for s in suggestions if s.get("category") == "long_term")
            },
            "current_state": {
                "total_time_hours": current_total_time,
                "total_time_minutes": current_total_time * 60,
                "total_cost": current_total_cost,
                "resource_count": len(process.resources),
                "task_count": len(process.tasks),
                "execution_mode": "sequential"
            },
            "optimized_state": {
                "total_time_hours": optimized_total_time,
                "total_time_minutes": optimized_total_time * 60,
                "total_cost": optimized_total_cost,
                "time_saved_hours": time_saved,
                "time_saved_minutes": time_saved * 60,
                "time_saved_percentage": time_saved_percentage,
                "execution_mode": "parallel" if parallel_tasks else "sequential"
            },
            "suggestions": suggestions,
            "task_assignments": task_assignments,
            "network_diagram": network_diagram,
            "parallel_execution": {
                "enabled": len(parallel_tasks) > 0,
                "execution_pattern": execution_pattern_analysis,
                "parallel_groups": parallel_tasks,
                "sequential_tasks": sequential_tasks,
                "total_parallel_tasks": sum(pt["task_count"] for pt in parallel_tasks),
                "total_sequential_tasks": len(sequential_tasks),
                "execution_steps": parallel_execution_steps,
                "total_steps": len(parallel_execution_steps)
            },
            "constraints": {
                "dependencies": [],
                "resource_limitations": [f"{r.name}: {r.max_hours_per_day} hours/day" for r in process.resources if hasattr(r, 'max_hours_per_day')],
                "skill_requirements": [f"{t.name} requires {', '.join(s.name for s in t.required_skills)}" for t in process.tasks if t.required_skills]
            },
            "risks": [
                {
                    "risk": "Quality Degradation",
                    "probability": "Medium",
                    "impact": "High",
                    "mitigation": "Implement quality checkpoints and random audits"
                },
                {
                    "risk": "Staff Resistance to Change",
                    "probability": "High",
                    "impact": "Medium",
                    "mitigation": "Involve staff in design, provide training, gradual rollout"
                },
                {
                    "risk": "Coordination Failures",
                    "probability": "Low",
                    "impact": "High",
                    "mitigation": "Regular sync meetings and clear communication protocols"
                }
            ],
            "improvements": _build_improvements_section(
                time_saved_percentage=time_saved_percentage,
                current_total_cost=current_total_cost,
                optimized_total_cost=optimized_total_cost,
                job_resolution_cost_summary=job_resolution_cost_summary,
                multi_job_resolutions=multi_job_resolutions,
                cost_optimization_result=cost_optimization_result
            )
        }
        
        # Add patient journey data for healthcare processes
        if patient_journey:
            response["patient_journey"] = patient_journey
        
        # Add multi-job resolution data
        # This shows how tasks with multiple jobs were resolved to 1:1 relationships
        if multi_job_resolutions:
            # Calculate summary statistics
            best_fit_count = sum(1 for r in multi_job_resolutions if r.get('resolution') == 'best_fit')
            split_count = sum(1 for r in multi_job_resolutions if r.get('resolution') == 'split')
            single_job_count = sum(1 for r in multi_job_resolutions if r.get('resolution') == 'single_job')
            total_sub_tasks = sum(len(r.get('sub_tasks', [])) for r in multi_job_resolutions)
            total_removed_jobs = sum(len(r.get('removed_jobs', [])) for r in multi_job_resolutions)
            
            # Calculate cost savings from job resolution (removing redundant parallel jobs)
            cost_before = job_resolution_cost_summary.get('total_before_cost', 0) if job_resolution_cost_summary else 0
            cost_after = job_resolution_cost_summary.get('total_after_cost', 0) if job_resolution_cost_summary else 0
            cost_savings = job_resolution_cost_summary.get('total_savings', 0) if job_resolution_cost_summary else 0
            savings_pct = job_resolution_cost_summary.get('savings_percentage', 0) if job_resolution_cost_summary else 0
            
            response["job_resolution"] = {
                "enabled": True,
                "description": "Multi-job tasks have been resolved to maintain 1:1 task-job relationship",
                "summary": {
                    "total_tasks_analyzed": len(multi_job_resolutions),
                    "single_job_tasks": single_job_count,
                    "best_fit_resolved": best_fit_count,
                    "split_into_subtasks": split_count,
                    "total_sub_tasks_created": total_sub_tasks,
                    "total_jobs_removed": total_removed_jobs,
                    "best_fit_threshold": "90%"
                },
                "cost_savings": {
                    "before_resolution_cost": cost_before,
                    "after_resolution_cost": cost_after,
                    "total_savings": cost_savings,
                    "savings_percentage": savings_pct,
                    "calculation_method": "Sum of (all assigned jobs hourly rates × task duration) vs (kept job hourly rate × task duration)"
                },
                "resolutions": multi_job_resolutions
            }
        else:
            response["job_resolution"] = {
                "enabled": True,
                "description": "No multi-job tasks found - all tasks already have 1:1 job relationship",
                "summary": {
                    "total_tasks_analyzed": 0,
                    "single_job_tasks": 0,
                    "best_fit_resolved": 0,
                    "split_into_subtasks": 0,
                    "total_sub_tasks_created": 0,
                    "total_jobs_removed": 0,
                    "best_fit_threshold": "90%"
                },
                "cost_savings": {
                    "before_resolution_cost": 0,
                    "after_resolution_cost": 0,
                    "total_savings": 0,
                    "savings_percentage": 0,
                    "calculation_method": "Sum of (all assigned jobs hourly rates × task duration) vs (kept job hourly rate × task duration)"
                },
                "resolutions": []
            }
        
        # Add cost optimization results
        # Shows jobs replaced with cheaper alternatives that have the same skills
        if cost_optimization_result:
            response["cost_optimization"] = {
                "enabled": True,
                "description": "Jobs have been replaced with cheaper alternatives that meet skill requirements",
                "summary": {
                    "original_total_cost": cost_optimization_result.get("original_total_cost", 0),
                    "optimized_total_cost": cost_optimization_result.get("optimized_total_cost", 0),
                    "total_savings": cost_optimization_result.get("total_savings", 0),
                    "savings_percentage": cost_optimization_result.get("savings_percentage", 0),
                    "tasks_analyzed": cost_optimization_result.get("tasks_analyzed", 0),
                    "tasks_optimized": cost_optimization_result.get("tasks_optimized", 0),
                    "skill_match_threshold": "90%"
                },
                "replacements": cost_optimization_result.get("replacements", [])
            }
        else:
            response["cost_optimization"] = {
                "enabled": True,
                "description": "No cheaper job alternatives found - current assignments are already optimal",
                "summary": {
                    "original_total_cost": 0,
                    "optimized_total_cost": 0,
                    "total_savings": 0,
                    "savings_percentage": 0,
                    "tasks_analyzed": 0,
                    "tasks_optimized": 0,
                    "skill_match_threshold": "90%"
                },
                "replacements": []
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in optimize_cms_process_json: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )


@app.get("/cms/whatif/{process_id}")
async def get_whatif_analysis_data(process_id: int, authorization: Optional[str] = Header(None)):
    """
    Get What-if Analysis data for a process from CMS.
    
    Returns:
    - Best optimized scenario (cost, time, quality, resource allocation)
    - Manual constraints for user adjustment:
      - Resources: hourly rate, max hours/day
      - Tasks: name, duration, priority
      - Preferences: parallel execution settings
    
    This endpoint provides the initial data structure for the What-if Analysis frontend page.
    """
    # Use provided token or authenticate dynamically
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
    
    # CMSClient will authenticate automatically if token is None
    client = CMSClient(base_url=DEFAULT_CMS_URL, bearer_token=token)
    transformer = CMSDataTransformer()
    
    try:
        # Fetch process from CMS
        cms_data = await asyncio.to_thread(client.get_process_with_relations, process_id)
        if not cms_data:
            raise HTTPException(status_code=404, detail=f"Process {process_id} not found in CMS")
        
        # Store original process data for comparison
        process_name = cms_data.get("process_name", "")
        
        # Fetch jobs with their real skills from CMS
        jobs_with_skills = {}
        all_jobs_map = {}
        try:
            jobs_with_skills = await asyncio.to_thread(client.get_jobs_for_process, cms_data)
            all_jobs_map = await asyncio.to_thread(client.get_all_jobs_map_with_skills)
        except Exception as e:
            print(f"[WARNING] Could not fetch jobs with skills: {e}")
        
        # Resolve multi-job tasks BEFORE transformation
        resolver = MultiJobResolver(best_fit_threshold=0.90, jobs_with_skills=jobs_with_skills)
        resolved_cms_data = resolver.resolve_process(cms_data)
        
        # Extract resolution details
        multi_job_resolutions = resolved_cms_data.pop('_multi_job_resolutions', [])
        job_resolution_cost_summary = resolved_cms_data.pop('_job_resolution_cost_summary', None)
        
        # Cost optimization
        cost_optimization_result = None
        cost_optimization_savings = 0.0
        if all_jobs_map:
            try:
                resolved_cms_data, cost_opt_result = optimize_process_cost(
                    resolved_cms_data, 
                    all_jobs_map, 
                    skill_match_threshold=0.90
                )
                cost_optimization_result = resolved_cms_data.pop('_cost_optimization', None)
                cost_optimization_savings = cost_opt_result.total_savings if cost_opt_result else 0.0
            except Exception as e:
                print(f"[WARNING] Cost optimization failed: {e}")
        
        # Transform to agent format
        try:
            agent_format = transformer.transform_process(resolved_cms_data)
        except ProcessValidationError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": e.error_code,
                    "message": e.message,
                    "process_id": process_id
                }
            )
        
        # Run optimization using IntelligentOptimizer
        from process_optimization_agent import ProcessIntelligence, ProcessType
        from process_optimization_agent.Optimization.models import Process, Task, Resource, Skill, SkillLevel
        
        # Convert agent_format to Process object
        process = Process(
            id=agent_format.get("id", str(process_id)),
            name=agent_format.get("process_name", ""),
            description=agent_format.get("description", ""),
            tasks=[],
            resources=[]
        )
        
        # Add tasks
        for task_data in agent_format.get("tasks", []):
            required_skills = []
            for skill in task_data.get("required_skills", []):
                if isinstance(skill, dict):
                    skill_name = skill.get("name", skill.get("skill_name", ""))
                else:
                    skill_name = str(skill)
                if skill_name:
                    required_skills.append(Skill(name=skill_name, level=SkillLevel.INTERMEDIATE))
            
            task = Task(
                id=task_data["id"],
                name=task_data["name"],
                description=task_data.get("description", ""),
                duration_hours=task_data["duration_hours"],
                required_skills=required_skills
            )
            process.tasks.append(task)
        
        # Calculate total required hours per resource skill
        total_hours_by_skill = {}
        for task in process.tasks:
            for skill in task.required_skills:
                skill_key = skill.name.lower().strip()
                total_hours_by_skill[skill_key] = total_hours_by_skill.get(skill_key, 0) + task.duration_hours
        
        # Add resources
        for resource_data in agent_format.get("resources", []):
            skills = []
            for skill in resource_data.get("skills", []):
                if isinstance(skill, dict):
                    skill_name = skill.get("name", skill.get("skill_name", ""))
                else:
                    skill_name = str(skill)
                if skill_name:
                    skills.append(Skill(name=skill_name, level=SkillLevel.INTERMEDIATE))
            
            max_required_hours = 160.0
            for skill in skills:
                skill_key = skill.name.lower().strip()
                if skill_key in total_hours_by_skill:
                    max_required_hours = max(max_required_hours, total_hours_by_skill[skill_key] * 1.2)
            
            resource = Resource(
                id=resource_data["id"],
                name=resource_data["name"],
                hourly_rate=resource_data.get("hourly_rate", 0),
                skills=skills,
                total_available_hours=max_required_hours
            )
            process.resources.append(resource)
        
        # Run intelligent optimization
        intelligent_optimizer = IntelligentOptimizer()
        optimization_result = await asyncio.to_thread(intelligent_optimizer.optimize, process)
        
        # Calculate metrics
        schedule = optimization_result.schedule
        
        # Calculate optimized state from actual task assignments
        if schedule and schedule.entries:
            optimized_total_time = max(entry.end_hour for entry in schedule.entries)
            optimized_total_cost = sum(
                (entry.end_hour - entry.start_hour) * 
                next((r.hourly_rate for r in process.resources if r.id == entry.resource_id), 50)
                for entry in schedule.entries
            )
        else:
            optimized_total_time = sum(task.duration_hours for task in process.tasks)
            optimized_total_cost = 0.0
        
        # Calculate original cost
        job_resolution_savings = job_resolution_cost_summary.get('total_savings', 0) if job_resolution_cost_summary else 0.0
        total_savings = job_resolution_savings + cost_optimization_savings
        original_total_cost = optimized_total_cost + total_savings
        
        # Build resource allocation from schedule
        resource_allocation = []
        if schedule and schedule.entries:
            for entry in schedule.entries:
                task = next((t for t in process.tasks if t.id == entry.task_id), None)
                resource = next((r for r in process.resources if r.id == entry.resource_id), None)
                if task and resource:
                    resource_allocation.append({
                        "task": task.name,
                        "task_id": task.id,
                        "resource": resource.name,
                        "resource_id": resource.id,
                        "hours": round(entry.end_hour - entry.start_hour, 2),
                        "cost": round((entry.end_hour - entry.start_hour) * resource.hourly_rate, 2),
                        "start": round(entry.start_hour, 2),
                        "end": round(entry.end_hour, 2)
                    })
        
        # Calculate quality score (higher is better)
        # Based on skill matching and resource utilization
        quality_score = 88.0  # Default base score
        if schedule and schedule.entries:
            # Increase score based on parallel execution efficiency
            parallel_efficiency = (sum(task.duration_hours for task in process.tasks) / optimized_total_time) if optimized_total_time > 0 else 1.0
            quality_score = min(91.0, 85.0 + (parallel_efficiency * 5))
        
        # Build manual constraints - Resources tab
        resources_constraints = []
        for resource_data in agent_format.get("resources", []):
            resources_constraints.append({
                "resource_id": resource_data["id"],
                "name": resource_data["name"],
                "hourly_rate": resource_data.get("hourly_rate", 0),
                "max_hours_per_day": resource_data.get("max_hours_per_day", 8),
                "skills": [
                    skill.get("name") if isinstance(skill, dict) else str(skill)
                    for skill in resource_data.get("skills", [])
                ]
            })
        
        # Build manual constraints - Tasks tab
        tasks_constraints = []
        # Get original task order from CMS
        process_tasks = cms_data.get("process_task", [])
        
        for idx, task_data in enumerate(agent_format.get("tasks", [])):
            # Find corresponding CMS task to get original order and priority
            cms_task_info = next(
                (pt for pt in process_tasks if str(pt.get("task", {}).get("task_id")) == str(task_data["id"])),
                None
            )
            
            original_order = cms_task_info.get("order", idx + 1) if cms_task_info else idx + 1
            
            tasks_constraints.append({
                "task_id": task_data["id"],
                "name": task_data["name"],
                "duration_minutes": int(task_data["duration_hours"] * 60),
                "duration_hours": task_data["duration_hours"],
                "order": original_order,
                "priority": "Normal",  # Default priority
                "allow_parallel": True,  # Default to allow parallel execution
                "required_skills": [
                    skill.get("name") if isinstance(skill, dict) else str(skill)
                    for skill in task_data.get("required_skills", [])
                ]
            })
        
        # Sort tasks by original order
        tasks_constraints.sort(key=lambda x: x["order"])
        
        # Build preferences constraints
        preferences_constraints = {
            "time_priority": 33,  # 0-100 slider (default middle)
            "cost_priority": 33,  # 0-100 slider (default middle)
            "quality_priority": 34,  # 0-100 slider (default middle, total = 100)
            "allow_parallel_execution": True,
            "max_parallel_tasks": len(process.tasks)
        }
        
        # Build the response in CMS expected format
        response = {
            "scenario": {
                "assignments": resource_allocation,
                "constraints": {
                    "resources": resources_constraints,
                    "tasks": tasks_constraints,
                    "preferences": preferences_constraints
                },
                "original": {
                    "duration_hours": round(sum(task.duration_hours for task in process.tasks), 2),
                    "duration_minutes": int(sum(task.duration_hours for task in process.tasks) * 60),
                    "total_cost": round(original_total_cost, 2),
                    "quality_score": 88.0,
                    "resource_utilization": 85.0
                }
            },
            "metrics": {
                "total_time_hours": round(optimized_total_time, 2),
                "total_time_minutes": int(optimized_total_time * 60),
                "total_time_days": round(optimized_total_time / 24, 2),
                "total_cost": round(optimized_total_cost, 2),
                "quality_score": round(quality_score, 1),
                "resource_utilization": 85.0
            },
            "metadata": {
                "process_id": process_id,
                "process_name": process_name,
                "optimization_timestamp": datetime.now().isoformat(),
                "optimization_engine": "IntelligentOptimizer",
                "total_tasks": len(process.tasks),
                "total_resources": len(process.resources)
            }
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_whatif_analysis_data: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch what-if analysis data: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("API.main:app", host="0.0.0.0", port=8000, reload=True)
