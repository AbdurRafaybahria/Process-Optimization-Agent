# Directory Structure

## Overview
The Process Optimization Agent is organized into a clean, modular structure with domain-specific folders for better maintainability.

## Root Structure

```
Process-Optimization-Agent/
├── API/                          # FastAPI application
│   ├── main.py                   # Main API endpoints
│   └── ...
│
├── process_optimization_agent/   # Core optimization package
│   ├── healthcare/               # Healthcare domain
│   │   ├── healthcare_models.py
│   │   ├── healthcare_optimizer.py
│   │   └── __init__.py
│   │
│   ├── manufacturing/            # Manufacturing domain
│   │   ├── manufacturing_models.py
│   │   ├── manufacturing_optimizer.py
│   │   ├── manufacturing_viz_helpers.py
│   │   └── __init__.py
│   │
│   ├── insurance/                # Insurance domain
│   │   ├── insurance_models.py
│   │   ├── insurance_optimizer.py
│   │   └── __init__.py
│   │
│   ├── models.py                 # Core data models
│   ├── optimizers.py             # Base optimizers
│   ├── intelligent_optimizer.py  # Main intelligent optimizer
│   ├── cms_transformer.py        # CMS data transformation
│   ├── process_intelligence.py   # Process type detection
│   └── ...
│
├── tests/                        # Test files
│   ├── test_process_detection.py
│   └── test_api_endpoints.py
│
├── scripts/                      # Utility scripts
│   ├── run_rl_optimizer.py
│   └── run_optimization.py
│
├── docs/                         # Documentation
│   ├── API_FLOW_COMPLETE.md
│   └── DIRECTORY_STRUCTURE.md
│
├── examples/                     # Example process JSON files
│   ├── customer_service_support.json
│   ├── premium_collection_billing.json
│   └── ...
│
├── visualization_outputs/        # Generated visualizations
│   └── *.png
│
└── outputs/                      # Optimization results
    └── ...
```

## Domain Modules

### Healthcare (`process_optimization_agent/healthcare/`)
- **Purpose**: Patient journey optimization and healthcare workflows
- **Key Files**:
  - `healthcare_models.py`: Healthcare-specific data models
  - `healthcare_optimizer.py`: Patient-centric optimization logic

### Manufacturing (`process_optimization_agent/manufacturing/`)
- **Purpose**: Production process optimization and throughput maximization
- **Key Files**:
  - `manufacturing_models.py`: Manufacturing-specific data models
  - `manufacturing_optimizer.py`: Production-centric optimization logic
  - `manufacturing_viz_helpers.py`: Manufacturing visualization utilities

### Insurance (`process_optimization_agent/insurance/`)
- **Purpose**: Insurance process optimization with scenario-based strategies
- **Key Files**:
  - `insurance_models.py`: Insurance-specific data models (12 scenarios)
  - `insurance_optimizer.py`: Insurance-centric optimization with fallback handling

## Import Structure

### From External Code
```python
# Import domain-specific optimizers
from process_optimization_agent.healthcare import HealthcareOptimizer
from process_optimization_agent.manufacturing import ManufacturingOptimizer
from process_optimization_agent.insurance import InsuranceProcessOptimizer

# Import core components
from process_optimization_agent import IntelligentOptimizer, Process, Task
```

### Within Domain Modules
```python
# Import from parent package
from ..models import Process, Task, Resource

# Import from same domain
from .healthcare_models import HealthcareMetrics
```

## Key Features

### ✅ Clean Separation
- Each domain has its own folder
- Clear separation between core and domain-specific code
- Tests and scripts in dedicated folders

### ✅ Modular Design
- Each domain is a Python package with `__init__.py`
- Easy to add new domains
- Independent development of domain logic

### ✅ Maintainability
- Documentation in `docs/` folder
- Examples in `examples/` folder
- Clear naming conventions

## Adding a New Domain

1. Create folder: `process_optimization_agent/new_domain/`
2. Add files:
   - `new_domain_models.py`
   - `new_domain_optimizer.py`
   - `__init__.py`
3. Update `process_optimization_agent/__init__.py`:
   ```python
   from .new_domain.new_domain_optimizer import NewDomainOptimizer
   ```
4. Update `intelligent_optimizer.py` to include new domain logic

## Validation & Error Handling

The system includes comprehensive validation:
- **No tasks**: Requires at least 2 tasks
- **No resources**: Requires at least 1 resource
- **Negative duration**: All durations must be positive
- **Negative/zero hourly rate**: All rates must be positive
- **Tasks without skills**: All tasks must have required skills

Validation errors return HTTP 400 with specific error codes and messages.

## Visualization Outputs

All generated visualizations are stored in `visualization_outputs/`:
- `insurance_summary_{id}.png` - Summary charts
- `insurance_allocation_{id}.png` - Resource allocation charts
- Automatically cleared on server restart
