"""
Banking-specific models and enums for Process Optimization Agent
Implements FR1-FR7: Banking process detection, classification, and constraints
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum
from datetime import datetime


class BankingProcessType(Enum):
    """Classification of banking processes (FR4)"""
    LOAN_APPROVAL = "loan_approval"
    ACCOUNT_OPENING = "account_opening"
    FUND_TRANSFER = "fund_transfer"
    CREDIT_CARD_APPLICATION = "credit_card_application"
    MORTGAGE_PROCESSING = "mortgage_processing"
    KYC_VERIFICATION = "kyc_verification"
    FRAUD_INVESTIGATION = "fraud_investigation"
    UNKNOWN = "unknown"


class ProcessStage(Enum):
    """Banking process stages (FR2)"""
    DOCUMENT_VERIFICATION = "document_verification"
    CREDIT_CHECK = "credit_check"
    RISK_ASSESSMENT = "risk_assessment"
    APPROVAL = "approval"
    DISBURSEMENT = "disbursement"
    KYC_CHECK = "kyc_check"
    ANTI_FRAUD_CHECK = "anti_fraud_check"
    ACCOUNT_SETUP = "account_setup"
    FUND_VALIDATION = "fund_validation"
    TRANSACTION_PROCESSING = "transaction_processing"
    COMPLIANCE_CHECK = "compliance_check"
    CUSTOMER_NOTIFICATION = "customer_notification"


class ConditionType(Enum):
    """Types of business rule conditions (FR5)"""
    CREDIT_SCORE = "credit_score"
    INCOME_LEVEL = "income_level"
    DEBT_RATIO = "debt_ratio"
    EMPLOYMENT_STATUS = "employment_status"
    ACCOUNT_BALANCE = "account_balance"
    TRANSACTION_AMOUNT = "transaction_amount"
    RISK_SCORE = "risk_score"
    AGE = "age"
    CITIZENSHIP = "citizenship"
    CUSTOM = "custom"


class ConditionOperator(Enum):
    """Operators for condition evaluation"""
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    EQUAL = "=="
    NOT_EQUAL = "!="
    IN = "in"
    NOT_IN = "not_in"


class ConstraintType(Enum):
    """Types of compliance constraints (FR6)"""
    KYC_REQUIRED = "kyc_required"
    ANTI_FRAUD_REQUIRED = "anti_fraud_required"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    DUAL_APPROVAL = "dual_approval"
    AUDIT_TRAIL = "audit_trail"
    DATA_ENCRYPTION = "data_encryption"
    CANNOT_SKIP = "cannot_skip"
    CRITICAL_STEP = "critical_step"


@dataclass
class BusinessRule:
    """Represents a business rule condition (FR5)"""
    id: str
    name: str
    condition_type: ConditionType
    operator: ConditionOperator
    threshold_value: Any
    action_on_true: str  # e.g., "approve", "reject", "escalate"
    action_on_false: str
    description: str = ""
    
    def evaluate(self, actual_value: Any) -> bool:
        """Evaluate the condition"""
        try:
            if self.operator == ConditionOperator.LESS_THAN:
                return actual_value < self.threshold_value
            elif self.operator == ConditionOperator.LESS_EQUAL:
                return actual_value <= self.threshold_value
            elif self.operator == ConditionOperator.GREATER_THAN:
                return actual_value > self.threshold_value
            elif self.operator == ConditionOperator.GREATER_EQUAL:
                return actual_value >= self.threshold_value
            elif self.operator == ConditionOperator.EQUAL:
                return actual_value == self.threshold_value
            elif self.operator == ConditionOperator.NOT_EQUAL:
                return actual_value != self.threshold_value
            elif self.operator == ConditionOperator.IN:
                return actual_value in self.threshold_value
            elif self.operator == ConditionOperator.NOT_IN:
                return actual_value not in self.threshold_value
        except Exception as e:
            print(f"Error evaluating rule {self.id}: {e}")
            return False
        return False
    
    def get_action(self, actual_value: Any) -> str:
        """Get the action based on evaluation"""
        if self.evaluate(actual_value):
            return self.action_on_true
        return self.action_on_false


@dataclass
class ComplianceConstraint:
    """Represents a compliance constraint (FR6, FR7)"""
    id: str
    name: str
    constraint_type: ConstraintType
    description: str = ""
    applies_to_tasks: Set[str] = field(default_factory=set)  # Task IDs this constraint applies to
    is_mandatory: bool = True
    validation_rules: List[str] = field(default_factory=list)
    
    def validate_task(self, task_id: str) -> bool:
        """Check if constraint is satisfied for a task"""
        return task_id in self.applies_to_tasks


@dataclass
class TaskDependency:
    """Represents a dependency between tasks (FR3)"""
    source_task_id: str
    target_task_id: str
    dependency_type: str  # "sequential", "conditional", "parallel_allowed"
    condition: Optional[BusinessRule] = None
    description: str = ""
    
    def is_satisfied(self, completed_tasks: Set[str], context: Dict[str, Any] = None) -> bool:
        """Check if dependency is satisfied"""
        if self.dependency_type == "sequential":
            return self.source_task_id in completed_tasks
        elif self.dependency_type == "conditional":
            if self.source_task_id not in completed_tasks:
                return False
            if self.condition and context:
                # Check if condition is met
                condition_value = context.get(self.condition.condition_type.value)
                if condition_value is not None:
                    return self.condition.evaluate(condition_value)
            return True
        elif self.dependency_type == "parallel_allowed":
            return True  # Can run in parallel
        return False


@dataclass
class BankingProcess:
    """Extended process model for banking operations (FR1-FR7)"""
    id: str
    name: str
    description: str
    process_type: BankingProcessType
    stages: List[ProcessStage] = field(default_factory=list)
    business_rules: List[BusinessRule] = field(default_factory=list)
    compliance_constraints: List[ComplianceConstraint] = field(default_factory=list)
    task_dependencies: List[TaskDependency] = field(default_factory=list)
    critical_tasks: Set[str] = field(default_factory=set)  # Tasks that cannot be removed (FR7)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def classify_process_type(self, task_names: List[str]) -> BankingProcessType:
        """Automatically classify process type based on tasks (FR4)"""
        task_text = " ".join(task_names).lower()
        
        # Keyword-based classification
        if any(keyword in task_text for keyword in ["loan", "lending", "credit approval"]):
            return BankingProcessType.LOAN_APPROVAL
        elif any(keyword in task_text for keyword in ["account opening", "new account", "account setup"]):
            return BankingProcessType.ACCOUNT_OPENING
        elif any(keyword in task_text for keyword in ["transfer", "payment", "transaction"]):
            return BankingProcessType.FUND_TRANSFER
        elif any(keyword in task_text for keyword in ["credit card", "card application"]):
            return BankingProcessType.CREDIT_CARD_APPLICATION
        elif any(keyword in task_text for keyword in ["mortgage", "home loan"]):
            return BankingProcessType.MORTGAGE_PROCESSING
        elif any(keyword in task_text for keyword in ["kyc", "know your customer", "identity verification"]):
            return BankingProcessType.KYC_VERIFICATION
        elif any(keyword in task_text for keyword in ["fraud", "suspicious", "investigation"]):
            return BankingProcessType.FRAUD_INVESTIGATION
        
        return BankingProcessType.UNKNOWN
    
    def identify_stages(self, tasks: List[Any]) -> List[ProcessStage]:
        """Identify process stages from tasks (FR2)"""
        stages = []
        stage_keywords = {
            ProcessStage.DOCUMENT_VERIFICATION: ["document", "verification", "verify", "check documents"],
            ProcessStage.CREDIT_CHECK: ["credit check", "credit score", "credit history"],
            ProcessStage.RISK_ASSESSMENT: ["risk", "assessment", "evaluate risk"],
            ProcessStage.APPROVAL: ["approval", "approve", "decision"],
            ProcessStage.DISBURSEMENT: ["disbursement", "disburse", "fund release"],
            ProcessStage.KYC_CHECK: ["kyc", "know your customer", "identity"],
            ProcessStage.ANTI_FRAUD_CHECK: ["fraud", "anti-fraud", "fraud detection"],
            ProcessStage.ACCOUNT_SETUP: ["account setup", "create account", "account creation"],
            ProcessStage.FUND_VALIDATION: ["fund validation", "validate funds", "check balance"],
            ProcessStage.TRANSACTION_PROCESSING: ["transaction", "process payment", "execute transfer"],
            ProcessStage.COMPLIANCE_CHECK: ["compliance", "regulatory", "regulation check"],
            ProcessStage.CUSTOMER_NOTIFICATION: ["notify", "notification", "inform customer"]
        }
        
        for task in tasks:
            task_text = f"{task.name} {task.description}".lower()
            for stage, keywords in stage_keywords.items():
                if any(keyword in task_text for keyword in keywords):
                    if stage not in stages:
                        stages.append(stage)
        
        return stages
    
    def detect_dependencies(self, tasks: List[Any]) -> List[TaskDependency]:
        """Detect dependencies between tasks (FR3)"""
        dependencies = []
        
        # Define common dependency patterns in banking
        dependency_patterns = [
            ("credit_check", "approval", "sequential"),
            ("document_verification", "credit_check", "sequential"),
            ("kyc_check", "account_setup", "sequential"),
            ("anti_fraud_check", "transaction_processing", "sequential"),
            ("approval", "disbursement", "sequential"),
            ("risk_assessment", "approval", "sequential"),
            ("compliance_check", "approval", "sequential"),
        ]
        
        task_dict = {task.id: task for task in tasks}
        
        # Check for pattern-based dependencies
        for source_pattern, target_pattern, dep_type in dependency_patterns:
            source_tasks = [t for t in tasks if source_pattern in t.name.lower() or source_pattern in t.description.lower()]
            target_tasks = [t for t in tasks if target_pattern in t.name.lower() or target_pattern in t.description.lower()]
            
            for source_task in source_tasks:
                for target_task in target_tasks:
                    if source_task.id != target_task.id:
                        dependencies.append(TaskDependency(
                            source_task_id=source_task.id,
                            target_task_id=target_task.id,
                            dependency_type=dep_type,
                            description=f"{target_task.name} depends on {source_task.name}"
                        ))
        
        return dependencies
    
    def validate_process_integrity(self, scheduled_tasks: Set[str]) -> Tuple[bool, List[str]]:
        """Ensure critical tasks are not skipped (FR7)"""
        missing_critical = []
        for critical_task_id in self.critical_tasks:
            if critical_task_id not in scheduled_tasks:
                missing_critical.append(critical_task_id)
        
        is_valid = len(missing_critical) == 0
        return is_valid, missing_critical
    
    def get_applicable_constraints(self, task_id: str) -> List[ComplianceConstraint]:
        """Get all constraints applicable to a task (FR6)"""
        return [c for c in self.compliance_constraints if task_id in c.applies_to_tasks]
