# Complete API Flow Documentation

## ✅ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CMS Frontend                             │
│                  (React Application)                             │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP Request
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                               │
│                    (API/main.py)                                 │
│                                                                   │
│  Endpoints:                                                       │
│  • POST /cms/optimize/{process_id}                               │
│  • POST /cms/optimize/{process_id}/alloc_png                     │
│  • POST /cms/optimize/{process_id}/summary_png                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CMS Data Pipeline                             │
│                                                                   │
│  1. CMSClient.get_process_with_relations(process_id)            │
│     └─> Fetches raw CMS data                                    │
│                                                                   │
│  2. CMSDataTransformer.transform_process(cms_data)              │
│     └─> Converts CMS format to Agent format                     │
│                                                                   │
│  3. write_temp_process_json(agent_format)                       │
│     └─> Writes to temp JSON file                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              run_rl_optimizer.py (Subprocess)                    │
│                                                                   │
│  1. Load Process from JSON                                       │
│  2. **Process Type Detection** (NEW!)                           │
│     └─> ProcessIntelligence.detect_process_type()              │
│         • Healthcare (99% if "patient" found)                   │
│         • Manufacturing (for development/production)            │
│         • Banking & Academic DISABLED                           │
│                                                                   │
│  3. RL-Based Optimization                                        │
│     └─> RLBasedOptimizer.optimize(process)                     │
│                                                                   │
│  4. **Unified Visualization** (NEW!)                            │
│     ├─> visualizer.create_allocation_page()                    │
│     │   └─> Auto-routes to healthcare or manufacturing         │
│     └─> visualizer.create_summary_page()                       │
│         └─> Auto-routes to healthcare or manufacturing         │
│                                                                   │
│  5. Generate 2 PNG Files:                                        │
│     • {process_id}_alloc_charts_{timestamp}.png                 │
│     • {process_id}_summary_{timestamp}.png                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API Response                                  │
│                                                                   │
│  Returns JSON with:                                              │
│  • process_id                                                    │
│  • process_name                                                  │
│  • original_cms_data                                             │
│  • transformed_data                                              │
│  • optimization_results:                                         │
│    ├─ alloc_chart (base64 + path)                              │
│    └─ summary_chart (base64 + path)                            │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Complete Request Flow

### Step 1: CMS Data Retrieval
```python
# API/main.py line 248
cms_data = await asyncio.to_thread(client.get_process_with_relations, process_id)
```
**Input**: Process ID from CMS
**Output**: Raw CMS JSON with nested structure

### Step 2: Data Transformation
```python
# API/main.py line 253
agent_format = transformer.transform_process(cms_data)
```
**Input**: CMS format (process_task, jobTasks, etc.)
**Output**: Agent format (tasks, resources, dependencies)

### Step 3: Process Type Detection
```python
# run_rl_optimizer.py line 192-196
intelligence = ProcessIntelligence()
classification = intelligence.detect_process_type(process)
```
**Detection Logic**:
- ✅ **CRITICAL RULE**: If "patient" anywhere → Healthcare (99%)
- ✅ **Healthcare**: 75+ keywords (patient, doctor, medical, etc.)
- ✅ **Manufacturing**: 100+ keywords (development, production, API, etc.)
- ❌ **Banking**: DISABLED
- ❌ **Academic**: DISABLED

### Step 4: Optimization
```python
# run_rl_optimizer.py line 201
schedule = optimizer.optimize(process)
```
**Algorithm**: RL-Based Parallel Scheduling
**Output**: Optimized schedule with resource assignments

### Step 5: Visualization Generation
```python
# run_rl_optimizer.py line 551-555 (Allocation)
visualizer.create_allocation_page(
    process=process,
    schedule=schedule,
    process_type=classification.process_type.value,
    save_path=alloc_chart_out
)

# run_rl_optimizer.py line 808-813 (Summary)
visualizer.create_summary_page(
    process=process,
    schedule=schedule,
    process_type=classification.process_type.value,
    before_metrics=before_metrics,
    save_path=chart_out
)
```

**Automatic Routing**:
- **Healthcare** → `create_healthcare_allocation_page()` + `create_healthcare_summary_page()`
- **Manufacturing** → `create_manufacturing_allocation_page()` + `create_manufacturing_summary_page()`

### Step 6: Return Results
```python
# API/main.py line 262-280
return {
    "process_id": str(process_id),
    "optimization_results": {
        "alloc_chart": {"base64": alloc_b64, ...},
        "summary_chart": {"base64": summary_b64, ...}
    }
}
```

## 📊 Visualization Differences

### Healthcare Visualizations
**Allocation Page**:
- Resource → Task Timeline (end-to-end arrows)
- Time Utilization per Resource (minutes)
- Cost per Resource
- Parallel Task Groups

**Summary Page**:
- Patient Journey Timeline (line graph with green boxes)
- Cumulative time markers
- Summary table with patient metrics

### Manufacturing Visualizations
**Allocation Page**:
- Resource → Task Timeline (duration-based arrows)
- Time Utilization per Resource (bar chart)
- Cost per Resource (bar chart)
- Parallel Task Groups

**Summary Page**:
- Duration Comparison (Before/After bar chart)
- Peak Resource Usage (Before/After bar chart)
- Total Cost (Before/After bar chart)
- Summary Table with improvements

## 🎯 Key Features

### 1. Automatic Process Detection
- ✅ **Patient keyword** = Automatic Healthcare (99% confidence)
- ✅ **Expanded dictionaries**: 75+ healthcare terms, 100+ manufacturing terms
- ✅ **Only 2 scenarios**: Healthcare and Manufacturing (Banking/Academic disabled)

### 2. Unified Visualization Interface
```python
# Single method call, auto-detects and routes
visualizer.create_allocation_page(process, schedule, process_type)
visualizer.create_summary_page(process, schedule, process_type, before_metrics)
```

### 3. Cost Accuracy
- ✅ Uses actual `schedule.total_cost` from optimized schedule
- ✅ Matches between graph and table
- ✅ Calculates from actual resource hourly rates

### 4. Text Wrapping
- ✅ Task names wrap at 20 characters per line
- ✅ No text overflow outside boxes
- ✅ Increased box height to accommodate multi-line text

## 🔧 API Endpoints

### 1. Full Optimization with JSON Response
```http
POST /cms/optimize/{process_id}
Authorization: Bearer {token}
```
**Returns**: Complete JSON with base64-encoded PNGs

### 2. Allocation Chart Only
```http
POST /cms/optimize/{process_id}/alloc_png
Authorization: Bearer {token}
```
**Returns**: PNG file directly

### 3. Summary Chart Only
```http
POST /cms/optimize/{process_id}/summary_png
Authorization: Bearer {token}
```
**Returns**: PNG file directly

## ✅ System Status

| Component | Status | Notes |
|-----------|--------|-------|
| CMS Integration | ✅ Working | Fetches and transforms data |
| Process Detection | ✅ Enhanced | 99% for patient, expanded dictionaries |
| Healthcare Viz | ✅ Working | Patient journey + allocation |
| Manufacturing Viz | ✅ Working | Before/after + allocation |
| Cost Calculation | ✅ Fixed | Uses actual schedule costs |
| Text Wrapping | ✅ Fixed | 20 chars/line, no overflow |
| API Endpoints | ✅ Working | Returns 2 PNGs in base64 |
| Banking Scenario | ❌ Disabled | Only Healthcare & Manufacturing |

## 🚀 Testing

### Test Healthcare Process:
```bash
# Via API
curl -X POST "http://localhost:8000/cms/optimize/10" \
  -H "Authorization: Bearer {token}"

# Direct (for testing only - requires transformed data)
python test_process_detection.py examples/patient_registration.json
```

### Test Manufacturing Process:
```bash
# Via API
curl -X POST "http://localhost:8000/cms/optimize/7" \
  -H "Authorization: Bearer {token}"

# Direct (for testing only - requires transformed data)
python test_process_detection.py examples/ecommerce_development.json
```

## 📝 Important Notes

1. **CMS Format Required**: API expects CMS format, transforms it automatically
2. **Process Detection**: Happens automatically in `run_rl_optimizer.py`
3. **2 PNG Files**: Always generates both allocation and summary
4. **No Manual Selection**: Process type detected automatically
5. **Patient = Healthcare**: Any mention of "patient" = 99% Healthcare confidence

## 🎉 Summary

**Everything is intact and working!**

✅ API receives CMS data
✅ Transforms to agent format  
✅ Detects process type (Healthcare/Manufacturing only)
✅ Optimizes with RL algorithm
✅ Generates 2 domain-specific PNGs
✅ Returns base64-encoded images to frontend

The system now provides **intelligent, automatic process detection** with **domain-specific visualizations** tailored to Healthcare or Manufacturing scenarios!
