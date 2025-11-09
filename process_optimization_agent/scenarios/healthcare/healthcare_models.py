"""
Healthcare-specific models and data structures
Separation of concerns for healthcare domain
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any
from enum import Enum


class HealthcareProcessType(Enum):
    """Types of healthcare processes"""
    OUTPATIENT_CONSULTATION = "outpatient_consultation"
    PATIENT_REGISTRATION = "patient_registration"
    EMERGENCY_CARE = "emergency_care"
    INPATIENT_ADMISSION = "inpatient_admission"
    DIAGNOSTIC_PROCEDURE = "diagnostic_procedure"
    SURGICAL_PROCEDURE = "surgical_procedure"
    PHARMACY_DISPENSING = "pharmacy_dispensing"
    LAB_TESTING = "lab_testing"


class PatientJourneyStage(Enum):
    """Stages in a patient's journey through healthcare process"""
    ARRIVAL = "arrival"
    REGISTRATION = "registration"
    TRIAGE = "triage"
    WAITING = "waiting"
    CONSULTATION = "consultation"
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    PRESCRIPTION = "prescription"
    DISCHARGE = "discharge"
    FOLLOW_UP = "follow_up"


@dataclass
class HealthcareMetrics:
    """Healthcare-specific performance metrics"""
    patient_waiting_time: float = 0.0
    patient_active_time: float = 0.0
    total_patient_journey_time: float = 0.0
    number_of_touchpoints: int = 0
    number_of_resource_changes: int = 0
    patient_satisfaction_score: float = 0.0
    clinical_quality_score: float = 0.0
    
    # Administrative metrics
    total_cost: float = 0.0
    resource_utilization: float = 0.0
    process_efficiency: float = 0.0


@dataclass
class PatientTouchpoint:
    """Represents a point where patient interacts with healthcare staff"""
    task_id: str
    task_name: str
    resource_id: str
    resource_name: str
    start_time: float
    duration: float
    interaction_type: str  # "direct", "passive", "admin"
    
    
@dataclass
class HealthcareProcess:
    """Extended process model for healthcare domain"""
    id: str
    name: str
    process_type: HealthcareProcessType
    description: str = ""
    
    # Patient journey tracking
    patient_touchpoints: List[PatientTouchpoint] = field(default_factory=list)
    journey_stages: List[PatientJourneyStage] = field(default_factory=list)
    
    # Healthcare-specific constraints
    max_patient_waiting_time: float = 120.0  # minutes
    min_clinical_quality_score: float = 0.8
    required_certifications: Set[str] = field(default_factory=set)
    
    # Regulatory compliance
    hipaa_compliant: bool = True
    requires_patient_consent: bool = True
    emergency_priority: bool = False
    
    def calculate_patient_satisfaction(self, waiting_time: float, total_time: float) -> float:
        """Calculate patient satisfaction based on waiting and total time"""
        if total_time == 0:
            return 0.0
        
        waiting_ratio = waiting_time / total_time
        
        # Lower waiting ratio = higher satisfaction
        if waiting_ratio < 0.2:
            return 1.0
        elif waiting_ratio < 0.4:
            return 0.8
        elif waiting_ratio < 0.6:
            return 0.6
        else:
            return 0.4
