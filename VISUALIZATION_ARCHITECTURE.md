# Process Optimization Visualization Architecture

## Separation of Concerns

The visualization system now implements **separation of concerns** by automatically detecting the process type and routing to the appropriate visualization strategy.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Visualizer Class                          │
├─────────────────────────────────────────────────────────────┤
│  Unified Interface (Auto-Detection)                         │
│  ├─ create_summary_page()                                   │
│  ├─ create_allocation_page()                                │
│  └─ _detect_process_type()                                  │
├─────────────────────────────────────────────────────────────┤
│  Healthcare Visualizations                                   │
│  ├─ create_healthcare_summary_page()                        │
│  │   └─ Patient Journey Timeline + Summary Table           │
│  └─ create_healthcare_allocation_page()                     │
│      ├─ Resource → Task Timeline                            │
│      ├─ Time Utilization per Resource                       │
│      ├─ Cost per Resource                                   │
│      └─ Parallel Task Groups                                │
├─────────────────────────────────────────────────────────────┤
│  Manufacturing Visualizations                                │
│  ├─ create_manufacturing_summary_page()                     │
│  │   ├─ Duration Comparison (Before/After)                  │
│  │   ├─ Peak Resource Usage Comparison                      │
│  │   ├─ Total Cost Comparison                               │
│  │   └─ Summary Table                                       │
│  └─ create_manufacturing_allocation_page()                  │
│      ├─ Resource → Task Timeline                            │
│      ├─ Time Utilization per Resource                       │
│      ├─ Cost per Resource                                   │
│      └─ Parallel Task Groups                                │
└─────────────────────────────────────────────────────────────┘
```

## Process Type Detection

The system automatically detects the process type based on:

### Healthcare Indicators
- **Keywords**: patient, medical, doctor, nurse, hospital, clinic, consultation, examination, treatment, diagnosis
- **Detection Scope**: Process name and task names
- **Result**: Routes to healthcare-specific visualizations

### Manufacturing/Production
- **Default**: Any process not matching healthcare indicators
- **Keywords** (optional): manufacturing, production, assembly, fabrication
- **Result**: Routes to manufacturing-specific visualizations

## Visualization Differences

### Healthcare Focus
1. **Summary Page**:
   - Patient Journey Timeline (line graph with cumulative time)
   - Task names inside green boxes
   - Time in minutes
   - Summary table with patient metrics

2. **Allocation Page**:
   - End-to-end arrows (same size)
   - Resource and task names as blue blocks
   - Time utilization in minutes
   - Cost per resource

### Manufacturing Focus
1. **Summary Page**:
   - Before/After Duration comparison (bar chart)
   - Before/After Peak Resource Usage (bar chart)
   - Before/After Total Cost (bar chart)
   - Comprehensive summary table with improvements

2. **Allocation Page**:
   - Resource → Task Timeline
   - Time Utilization per Resource
   - Cost per Resource
   - Parallel Task Groups visualization

## Usage Examples

### Automatic Detection
```python
visualizer = Visualizer()

# Auto-detects process type and creates appropriate visualization
visualizer.create_summary_page(process, schedule, save_path="summary.png")
visualizer.create_allocation_page(process, schedule, save_path="allocation.png")
```

### Explicit Type Specification
```python
# Force healthcare visualization
visualizer.create_summary_page(process, schedule, 
                               process_type="Healthcare",
                               save_path="healthcare_summary.png")

# Force manufacturing visualization
visualizer.create_summary_page(process, schedule,
                               process_type="Manufacturing",
                               before_metrics={'duration': 32, 'resources': 5, 'cost': 800},
                               save_path="manufacturing_summary.png")
```

### Direct Method Calls
```python
# Call specific visualization directly
visualizer.create_healthcare_summary_page(process, schedule, save_path="hc_summary.png")
visualizer.create_manufacturing_allocation_page(process, schedule, save_path="mfg_allocation.png")
```

## Key Features

### 1. Separation of Concerns
- Healthcare and manufacturing visualizations are completely separate
- Each has its own layout, charts, and focus areas
- No code duplication - shared utilities where appropriate

### 2. Automatic Routing
- Intelligent process type detection
- Seamless switching between visualization strategies
- Fallback to manufacturing for unknown types

### 3. Consistent Interface
- Same method names for both types
- Similar parameter structure
- Easy to extend with new process types

### 4. Healthcare Specifics
- Patient-centric visualizations
- Journey timeline with waiting times
- Minute-based time display
- Task names embedded in visualization

### 5. Manufacturing Specifics
- Before/After comparisons
- Optimization metrics highlighted
- Resource efficiency focus
- Parallel processing visualization

## File Structure

```
process_optimization_agent/
├── visualizer.py                    # Main visualization class
│   ├── Unified Interface            # Auto-detection and routing
│   ├── Healthcare Visualizations    # Patient journey, etc.
│   └── Manufacturing Visualizations # Before/after comparisons
└── manufacturing_viz_helpers.py     # Helper methods (reference)
```

## Extension Points

To add a new process type (e.g., "Logistics"):

1. **Add detection logic** in `_detect_process_type()`
2. **Create visualization methods**:
   - `create_logistics_summary_page()`
   - `create_logistics_allocation_page()`
3. **Update routing** in `create_summary_page()` and `create_allocation_page()`
4. **Implement specific charts** for logistics domain

## Benefits

✅ **Clean Architecture**: Each process type has its own visualization strategy
✅ **Easy Maintenance**: Changes to one type don't affect others
✅ **Extensible**: Easy to add new process types
✅ **User-Friendly**: Automatic detection requires no user input
✅ **Flexible**: Can override auto-detection when needed
✅ **Domain-Specific**: Each visualization optimized for its domain
