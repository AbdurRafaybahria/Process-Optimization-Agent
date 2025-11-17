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
        
        # Add resources
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
            
            resource = Resource(
                id=resource_data["id"],
                name=resource_data["name"],
                hourly_rate=resource_data.get("hourly_rate", 0),
                skills=skills
            )
            process.resources.append(resource)
        
        # Run intelligent optimization
        intelligent_optimizer = IntelligentOptimizer()
        optimization_result = await asyncio.to_thread(intelligent_optimizer.optimize, process)
        
        # Calculate metrics
        schedule = optimization_result.schedule
        
        # Calculate current state (sequential execution)
        current_total_time = sum(task.duration_hours for task in process.tasks)
        current_total_cost = sum(
            task.duration_hours * (
                next((r.hourly_rate for r in process.resources 
                     if any(ts.name in [rs.name for rs in r.skills] 
                           for ts in task.required_skills)), 50)
            )
            for task in process.tasks
        )
        
        # Calculate optimized state
        if schedule and schedule.entries:
            optimized_total_time = max(entry.end_hour for entry in schedule.entries)
            optimized_total_cost = sum(
                (entry.end_hour - entry.start_hour) * 
                next((r.hourly_rate for r in process.resources if r.id == entry.resource_id), 50)
                for entry in schedule.entries
            )
        else:
            optimized_total_time = current_total_time
            optimized_total_cost = current_total_cost
        
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
        
        # Identify parallel execution opportunities
        parallel_tasks = []
        if schedule and schedule.entries:
            # Group tasks by start time
            time_groups = {}
            for entry in schedule.entries:
                if entry.start_hour not in time_groups:
                    time_groups[entry.start_hour] = []
                time_groups[entry.start_hour].append(entry)
            
            # Find parallel execution points
            for start_time, entries in time_groups.items():
                if len(entries) > 1:
                    task_names = [next((t.name for t in process.tasks if t.id == e.task_id), "") for e in entries]
                    parallel_tasks.append({
                        "start_time": start_time,
                        "task_count": len(entries),
                        "tasks": task_names
                    })
        
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
        
        # Build the complete response
        response = {
            "process_id": str(process_id),
            "process_name": process.name,
            "company": agent_format.get("company", ""),
            "process_type": {
                "type": optimization_result.detected_type.name if hasattr(optimization_result, 'detected_type') else "GENERIC",
                "confidence": getattr(optimization_result, 'confidence', 0),
                "strategy": getattr(optimization_result, 'strategy', "")
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
            "parallel_execution": {
                "enabled": len(parallel_tasks) > 0,
                "parallel_groups": parallel_tasks,
                "total_parallel_tasks": sum(pt["task_count"] for pt in parallel_tasks)
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
            "improvements": {
                "time_efficiency": f"{time_saved_percentage:.1f}% faster",
                "cost_efficiency": f"${abs(optimized_total_cost - current_total_cost):.2f} {'increase' if optimized_total_cost > current_total_cost else 'decrease'}",
                "resource_utilization": "Improved through parallel execution",
                "process_flexibility": "Enhanced ability to handle variable workloads"
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in optimize_cms_process_json: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("API.main:app", host="0.0.0.0", port=8000, reload=True)
