"""
Insurance-specific models and data structures
Comprehensive coverage for all insurance process scenarios
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any
from enum import Enum


class InsuranceScenarioType(Enum):
    """Types of insurance process scenarios"""
    STANDARD_BILLING = "standard_billing"  # Bill → Verify → Submit → Reconcile
    PRE_AUTHORIZATION = "pre_authorization"  # Verify → Pre-Auth → Approval → Bill → Submit
    EMERGENCY_CARE = "emergency_care"  # Treat → Retroactive Verify → Bill → Submit
    DENIED_APPEALS = "denied_appeals"  # Analyze → Document → Appeal → Resubmit
    MULTI_PAYER = "multi_payer"  # Primary → Secondary → Tertiary → Reconcile
    SELF_PAY = "self_pay"  # Bill → Payment → Optional Insurance → Reimbursement
    WORKERS_COMP = "workers_comp"  # Document → Notify → Verify → Bill → Submit
    GOVERNMENT_INSURANCE = "government_insurance"  # Verify → Compliance → Bill → Submit (Medicare/Medicaid)
    PHARMACY_DME = "pharmacy_dme"  # Verify → Formulary → Dispense → Bill → Submit
    BUNDLED_PAYMENTS = "bundled_payments"  # Aggregate → Bundle → Submit → Risk Adjust
    UNKNOWN = "unknown"


class InsuranceTaskType(Enum):
    """Types of tasks in insurance processes"""
    BILL_GENERATION = "bill_generation"
    INSURANCE_VERIFICATION = "insurance_verification"
    PRE_AUTHORIZATION = "pre_authorization"
    CLAIM_SUBMISSION = "claim_submission"
    RECORD_KEEPING = "record_keeping"
    CLAIM_RECONCILIATION = "claim_reconciliation"
    DENIAL_ANALYSIS = "denial_analysis"
    APPEAL_PREPARATION = "appeal_preparation"
    PAYMENT_PROCESSING = "payment_processing"
    COMPLIANCE_CHECK = "compliance_check"
    DOCUMENTATION = "documentation"
    FOLLOW_UP = "follow_up"
    COB_DETERMINATION = "cob_determination"  # Coordination of Benefits
    FORMULARY_CHECK = "formulary_check"
    RISK_ADJUSTMENT = "risk_adjustment"


class PayerType(Enum):
    """Types of insurance payers"""
    COMMERCIAL = "commercial"
    MEDICARE = "medicare"
    MEDICAID = "medicaid"
    WORKERS_COMP = "workers_compensation"
    SELF_PAY = "self_pay"
    GOVERNMENT = "government"
    MIXED = "mixed"


class UrgencyLevel(Enum):
    """Urgency levels for insurance processing"""
    EMERGENCY = "emergency"  # Immediate processing required
    URGENT = "urgent"  # Same-day processing
    ROUTINE = "routine"  # Standard timeline
    SCHEDULED = "scheduled"  # Pre-planned procedures


class ComplexityLevel(Enum):
    """Complexity levels for insurance claims"""
    SIMPLE = "simple"  # Single payer, standard services
    MODERATE = "moderate"  # Pre-auth required, multiple services
    COMPLEX = "complex"  # Multiple payers, appeals, special cases
    HIGH_RISK = "high_risk"  # Experimental treatments, high-value claims


@dataclass
class InsuranceMetrics:
    """Insurance-specific performance metrics"""
    # Time metrics
    total_process_time: float = 0.0  # minutes
    claim_submission_time: float = 0.0
    verification_time: float = 0.0
    reconciliation_time: float = 0.0
    
    # Cost metrics
    total_labor_cost: float = 0.0
    cost_per_claim: float = 0.0
    
    # Efficiency metrics
    throughput_per_day: float = 0.0  # claims processed per day
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    bottleneck_resources: List[str] = field(default_factory=list)
    
    # Quality metrics
    claim_approval_rate: float = 0.0  # percentage
    denial_rate: float = 0.0  # percentage
    first_pass_yield: float = 0.0  # percentage approved on first submission
    error_rate: float = 0.0  # percentage
    
    # Capacity metrics
    daily_capacity: float = 0.0  # max claims per day
    current_utilization_percent: float = 0.0
    capacity_buffer_percent: float = 0.0
    
    # Optimization gains
    time_savings_percent: float = 0.0
    time_savings_minutes: float = 0.0
    cost_savings_dollars: float = 0.0
    capacity_increase_percent: float = 0.0
    
    # Before/After comparison
    before_process_time: float = 0.0
    after_process_time: float = 0.0
    before_cost: float = 0.0
    after_cost: float = 0.0


@dataclass
class InsuranceProcess:
    """Extended process model for insurance workflows"""
    scenario_type: InsuranceScenarioType
    payer_type: PayerType
    urgency_level: UrgencyLevel
    complexity_level: ComplexityLevel
    
    # Process characteristics
    requires_pre_auth: bool = False
    requires_compliance_check: bool = False
    is_multi_payer: bool = False
    is_appeal: bool = False
    is_emergency: bool = False
    
    # Payer-specific details
    payer_count: int = 1
    payer_names: List[str] = field(default_factory=list)
    
    # Financial details
    claim_amount: float = 0.0
    expected_reimbursement: float = 0.0
    
    # Metadata
    detection_confidence: float = 0.0
    detection_reasoning: List[str] = field(default_factory=list)


@dataclass
class TaskParallelizationOpportunity:
    """Represents an opportunity to parallelize tasks"""
    task_group: List[str]  # Task IDs that can run in parallel
    time_saved: float  # Minutes saved by parallelization
    dependencies_satisfied: bool
    resource_availability: bool
    recommendation: str


@dataclass
class BottleneckAnalysis:
    """Analysis of process bottlenecks"""
    resource_name: str
    resource_id: str
    total_workload_minutes: float
    utilization_percent: float
    tasks_assigned: List[str]
    impact_on_throughput: str  # "Critical", "High", "Medium", "Low"
    recommendations: List[str]


@dataclass
class OptimizationRecommendation:
    """Specific optimization recommendation"""
    priority: str  # "IMMEDIATE", "SHORT_TERM", "MEDIUM_TERM", "LONG_TERM"
    category: str  # "Process", "Resource", "Technology", "Training"
    title: str
    description: str
    expected_impact: str
    implementation_cost: float
    roi_months: float
    risk_level: str  # "Low", "Medium", "High"
    mitigation_strategies: List[str] = field(default_factory=list)


@dataclass
class InsuranceOptimizationResult:
    """Complete optimization result for insurance processes"""
    # Process identification
    scenario_type: InsuranceScenarioType
    confidence: float
    
    # Current state
    current_metrics: InsuranceMetrics
    
    # Optimized state
    optimized_metrics: InsuranceMetrics
    
    # Analysis
    bottlenecks: List[BottleneckAnalysis]
    parallelization_opportunities: List[TaskParallelizationOpportunity]
    
    # Recommendations
    recommendations: List[OptimizationRecommendation]
    
    # Implementation
    implementation_phases: List[Dict[str, Any]]
    risks: List[Dict[str, Any]]
    success_metrics: List[str]
    
    # Optimized schedule (with default value, must come after non-default fields)
    optimized_schedule: Any = None  # Schedule object with optimized task assignments
    user_involved: bool = False  # Whether user/patient is directly involved in the process
    
    # Visualization data
    task_flow_before: Dict[str, Any] = field(default_factory=dict)
    task_flow_after: Dict[str, Any] = field(default_factory=dict)
    resource_utilization_before: Dict[str, float] = field(default_factory=dict)
    resource_utilization_after: Dict[str, float] = field(default_factory=dict)
