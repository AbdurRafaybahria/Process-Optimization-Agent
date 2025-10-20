"""
Manufacturing/Production-specific models and data structures
Separation of concerns for manufacturing domain
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any
from enum import Enum


class ManufacturingProcessType(Enum):
    """Types of manufacturing/production processes"""
    ASSEMBLY_LINE = "assembly_line"
    BATCH_PRODUCTION = "batch_production"
    CONTINUOUS_PRODUCTION = "continuous_production"
    JOB_SHOP = "job_shop"
    SOFTWARE_DEVELOPMENT = "software_development"
    PRODUCT_DEVELOPMENT = "product_development"
    QUALITY_CONTROL = "quality_control"
    PACKAGING = "packaging"


class ProductionStage(Enum):
    """Stages in manufacturing/production process"""
    PLANNING = "planning"
    DESIGN = "design"
    PROCUREMENT = "procurement"
    FABRICATION = "fabrication"
    ASSEMBLY = "assembly"
    TESTING = "testing"
    QUALITY_ASSURANCE = "quality_assurance"
    PACKAGING = "packaging"
    SHIPPING = "shipping"


@dataclass
class ManufacturingMetrics:
    """Manufacturing-specific performance metrics"""
    cycle_time: float = 0.0  # Total time from start to finish
    throughput: float = 0.0  # Units per hour
    production_cost: float = 0.0
    resource_utilization: float = 0.0
    
    # Efficiency metrics
    time_efficiency: float = 0.0  # Actual time vs theoretical minimum
    cost_efficiency: float = 0.0
    
    # Parallelization metrics
    max_parallel_tasks: int = 0
    parallel_time_percentage: float = 0.0
    
    # Quality metrics
    defect_rate: float = 0.0
    rework_percentage: float = 0.0
    
    # Resource metrics
    idle_time_hours: float = 0.0
    total_resource_hours: float = 0.0
    workload_balance_score: float = 0.0


@dataclass
class ProductionTask:
    """Extended task model for manufacturing"""
    task_id: str
    task_name: str
    machine_required: Optional[str] = None
    setup_time: float = 0.0
    processing_time: float = 0.0
    teardown_time: float = 0.0
    batch_size: int = 1
    quality_check_required: bool = False
    
    
@dataclass
class ManufacturingProcess:
    """Extended process model for manufacturing domain"""
    id: str
    name: str
    process_type: ManufacturingProcessType
    description: str = ""
    
    # Production tracking
    production_stages: List[ProductionStage] = field(default_factory=list)
    production_tasks: List[ProductionTask] = field(default_factory=list)
    
    # Manufacturing-specific constraints
    max_cycle_time: float = 480.0  # minutes
    min_throughput: float = 1.0  # units per hour
    quality_threshold: float = 0.95
    
    # Resource constraints
    machine_availability: Dict[str, float] = field(default_factory=dict)
    shift_schedule: List[Dict[str, Any]] = field(default_factory=list)
    
    # Production goals
    target_output: int = 1
    minimize_cost: bool = True
    maximize_throughput: bool = True
    balance_workload: bool = True
    
    def calculate_throughput(self, cycle_time: float) -> float:
        """Calculate throughput based on cycle time"""
        if cycle_time == 0:
            return 0.0
        return 60.0 / cycle_time  # units per hour
    
    def calculate_efficiency(self, actual_time: float, theoretical_time: float) -> float:
        """Calculate time efficiency"""
        if actual_time == 0:
            return 0.0
        return min(1.0, theoretical_time / actual_time)
