"""
Banking Process Detection and Analysis Module
Implements FR1-FR4: Process detection, stage analysis, dependency detection, and classification
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from .models import Process, Task, Resource
from .banking_models import (
    BankingProcess, BankingProcessType, ProcessStage, 
    TaskDependency, BusinessRule, ComplianceConstraint,
    ConditionType, ConditionOperator, ConstraintType
)


class BankingProcessDetector:
    """Detects and analyzes banking processes (FR1, FR2, FR3, FR4)"""
    
    def __init__(self):
        self.process_patterns = self._initialize_patterns()
        self.stage_keywords = self._initialize_stage_keywords()
        self.dependency_rules = self._initialize_dependency_rules()
    
    def _initialize_patterns(self) -> Dict[BankingProcessType, List[str]]:
        """Initialize process type detection patterns"""
        return {
            BankingProcessType.LOAN_APPROVAL: [
                "loan", "lending", "credit approval", "loan application",
                "loan processing", "loan disbursement"
            ],
            BankingProcessType.ACCOUNT_OPENING: [
                "account opening", "new account", "account setup",
                "account creation", "open account"
            ],
            BankingProcessType.FUND_TRANSFER: [
                "transfer", "payment", "transaction", "wire transfer",
                "fund transfer", "money transfer"
            ],
            BankingProcessType.CREDIT_CARD_APPLICATION: [
                "credit card", "card application", "card approval",
                "credit card processing"
            ],
            BankingProcessType.MORTGAGE_PROCESSING: [
                "mortgage", "home loan", "property loan",
                "mortgage approval", "mortgage processing"
            ],
            BankingProcessType.KYC_VERIFICATION: [
                "kyc", "know your customer", "identity verification",
                "customer verification", "identity check"
            ],
            BankingProcessType.FRAUD_INVESTIGATION: [
                "fraud", "suspicious", "investigation",
                "fraud detection", "anti-fraud"
            ]
        }
    
    def _initialize_stage_keywords(self) -> Dict[ProcessStage, List[str]]:
        """Initialize stage detection keywords"""
        return {
            ProcessStage.DOCUMENT_VERIFICATION: [
                "document", "verification", "verify", "check documents",
                "document review", "validate documents"
            ],
            ProcessStage.CREDIT_CHECK: [
                "credit check", "credit score", "credit history",
                "credit report", "credit assessment"
            ],
            ProcessStage.RISK_ASSESSMENT: [
                "risk", "assessment", "evaluate risk", "risk analysis",
                "risk evaluation", "risk scoring"
            ],
            ProcessStage.APPROVAL: [
                "approval", "approve", "decision", "authorize",
                "authorization", "approve request"
            ],
            ProcessStage.DISBURSEMENT: [
                "disbursement", "disburse", "fund release",
                "release funds", "payout", "payment"
            ],
            ProcessStage.KYC_CHECK: [
                "kyc", "know your customer", "identity",
                "identity verification", "customer verification"
            ],
            ProcessStage.ANTI_FRAUD_CHECK: [
                "fraud", "anti-fraud", "fraud detection",
                "fraud check", "suspicious activity"
            ],
            ProcessStage.ACCOUNT_SETUP: [
                "account setup", "create account", "account creation",
                "setup account", "initialize account"
            ],
            ProcessStage.FUND_VALIDATION: [
                "fund validation", "validate funds", "check balance",
                "verify funds", "balance check"
            ],
            ProcessStage.TRANSACTION_PROCESSING: [
                "transaction", "process payment", "execute transfer",
                "process transaction", "payment processing"
            ],
            ProcessStage.COMPLIANCE_CHECK: [
                "compliance", "regulatory", "regulation check",
                "compliance verification", "regulatory compliance"
            ],
            ProcessStage.CUSTOMER_NOTIFICATION: [
                "notify", "notification", "inform customer",
                "customer notification", "send notification"
            ]
        }
    
    def _initialize_dependency_rules(self) -> List[Tuple[str, str, str]]:
        """Initialize dependency detection rules (source_pattern, target_pattern, type)"""
        return [
            # Sequential dependencies - must complete source before target
            ("document", "credit", "sequential"),
            ("kyc", "account", "sequential"),
            ("kyc", "approval", "sequential"),
            ("credit", "approval", "sequential"),
            ("risk", "approval", "sequential"),
            ("compliance", "approval", "sequential"),
            ("anti-fraud", "transaction", "sequential"),
            ("anti-fraud", "approval", "sequential"),
            ("approval", "disbursement", "sequential"),
            ("approval", "account setup", "sequential"),
            ("fund validation", "transaction", "sequential"),
            ("verification", "approval", "sequential"),
            
            # Parallel allowed - can run simultaneously
            ("document", "kyc", "parallel_allowed"),
            ("credit", "risk", "parallel_allowed"),
        ]
    
    def detect_process(self, process: Process) -> BankingProcess:
        """
        Detect and analyze banking process (FR1)
        Returns a BankingProcess with full analysis
        """
        # FR4: Classify process type
        process_type = self._classify_process_type(process)
        
        # FR2: Identify stages
        stages = self._identify_stages(process.tasks)
        
        # FR3: Detect dependencies
        dependencies = self._detect_dependencies(process.tasks)
        
        # Create banking process
        banking_process = BankingProcess(
            id=process.id,
            name=process.name,
            description=process.description,
            process_type=process_type,
            stages=stages,
            task_dependencies=dependencies,
            metadata={"original_process": process}
        )
        
        # Add default business rules and constraints based on process type
        banking_process.business_rules = self._generate_business_rules(process_type)
        banking_process.compliance_constraints = self._generate_compliance_constraints(
            process_type, process.tasks
        )
        banking_process.critical_tasks = self._identify_critical_tasks(process.tasks, process_type)
        
        return banking_process
    
    def _classify_process_type(self, process: Process) -> BankingProcessType:
        """Classify the banking process type (FR4)"""
        # Combine all task names and descriptions
        text = f"{process.name} {process.description} "
        text += " ".join([f"{t.name} {t.description}" for t in process.tasks])
        text = text.lower()
        
        # Score each process type
        scores = {}
        for process_type, keywords in self.process_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                scores[process_type] = score
        
        # Return highest scoring type or UNKNOWN
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return BankingProcessType.UNKNOWN
    
    def _identify_stages(self, tasks: List[Task]) -> List[ProcessStage]:
        """Identify process stages from tasks (FR2)"""
        identified_stages = []
        
        for task in tasks:
            task_text = f"{task.name} {task.description}".lower()
            
            for stage, keywords in self.stage_keywords.items():
                if any(keyword in task_text for keyword in keywords):
                    if stage not in identified_stages:
                        identified_stages.append(stage)
        
        # Sort stages in typical banking process order
        stage_order = list(ProcessStage)
        identified_stages.sort(key=lambda s: stage_order.index(s))
        
        return identified_stages
    
    def _detect_dependencies(self, tasks: List[Task]) -> List[TaskDependency]:
        """Detect dependencies and conditions between tasks (FR3)"""
        dependencies = []
        task_dict = {task.id: task for task in tasks}
        
        # First, preserve existing dependencies from tasks
        for task in tasks:
            if hasattr(task, 'dependencies') and task.dependencies:
                for dep_id in task.dependencies:
                    if dep_id in task_dict:
                        dependencies.append(TaskDependency(
                            source_task_id=dep_id,
                            target_task_id=task.id,
                            dependency_type="sequential",
                            description=f"{task.name} depends on {task_dict[dep_id].name}"
                        ))
        
        # Detect additional pattern-based dependencies
        for source_pattern, target_pattern, dep_type in self.dependency_rules:
            source_tasks = [
                t for t in tasks 
                if source_pattern in t.name.lower() or source_pattern in t.description.lower()
            ]
            target_tasks = [
                t for t in tasks 
                if target_pattern in t.name.lower() or target_pattern in t.description.lower()
            ]
            
            for source_task in source_tasks:
                for target_task in target_tasks:
                    if source_task.id != target_task.id:
                        # Check if dependency already exists
                        exists = any(
                            d.source_task_id == source_task.id and 
                            d.target_task_id == target_task.id
                            for d in dependencies
                        )
                        
                        if not exists:
                            dependencies.append(TaskDependency(
                                source_task_id=source_task.id,
                                target_task_id=target_task.id,
                                dependency_type=dep_type,
                                description=f"{target_task.name} depends on {source_task.name}"
                            ))
        
        return dependencies
    
    def _generate_business_rules(self, process_type: BankingProcessType) -> List[BusinessRule]:
        """Generate default business rules based on process type (FR5)"""
        rules = []
        
        if process_type == BankingProcessType.LOAN_APPROVAL:
            rules.extend([
                BusinessRule(
                    id="rule_credit_score",
                    name="Credit Score Check",
                    condition_type=ConditionType.CREDIT_SCORE,
                    operator=ConditionOperator.LESS_THAN,
                    threshold_value=600,
                    action_on_true="reject",
                    action_on_false="proceed",
                    description="Reject loan if credit score < 600"
                ),
                BusinessRule(
                    id="rule_debt_ratio",
                    name="Debt-to-Income Ratio Check",
                    condition_type=ConditionType.DEBT_RATIO,
                    operator=ConditionOperator.GREATER_THAN,
                    threshold_value=0.43,
                    action_on_true="reject",
                    action_on_false="proceed",
                    description="Reject if debt-to-income ratio > 43%"
                ),
                BusinessRule(
                    id="rule_income",
                    name="Minimum Income Check",
                    condition_type=ConditionType.INCOME_LEVEL,
                    operator=ConditionOperator.LESS_THAN,
                    threshold_value=30000,
                    action_on_true="escalate",
                    action_on_false="proceed",
                    description="Escalate if annual income < $30,000"
                )
            ])
        
        elif process_type == BankingProcessType.ACCOUNT_OPENING:
            rules.extend([
                BusinessRule(
                    id="rule_age",
                    name="Age Verification",
                    condition_type=ConditionType.AGE,
                    operator=ConditionOperator.LESS_THAN,
                    threshold_value=18,
                    action_on_true="reject",
                    action_on_false="proceed",
                    description="Reject if age < 18"
                ),
                BusinessRule(
                    id="rule_citizenship",
                    name="Citizenship Check",
                    condition_type=ConditionType.CITIZENSHIP,
                    operator=ConditionOperator.NOT_IN,
                    threshold_value=["US", "Permanent Resident"],
                    action_on_true="reject",
                    action_on_false="proceed",
                    description="Reject if not US citizen or permanent resident"
                )
            ])
        
        elif process_type == BankingProcessType.FUND_TRANSFER:
            rules.extend([
                BusinessRule(
                    id="rule_balance",
                    name="Sufficient Balance Check",
                    condition_type=ConditionType.ACCOUNT_BALANCE,
                    operator=ConditionOperator.LESS_THAN,
                    threshold_value=0,
                    action_on_true="reject",
                    action_on_false="proceed",
                    description="Reject if insufficient balance"
                ),
                BusinessRule(
                    id="rule_large_transaction",
                    name="Large Transaction Check",
                    condition_type=ConditionType.TRANSACTION_AMOUNT,
                    operator=ConditionOperator.GREATER_THAN,
                    threshold_value=10000,
                    action_on_true="escalate",
                    action_on_false="proceed",
                    description="Escalate if transaction > $10,000"
                )
            ])
        
        return rules
    
    def _generate_compliance_constraints(
        self, 
        process_type: BankingProcessType, 
        tasks: List[Task]
    ) -> List[ComplianceConstraint]:
        """Generate compliance constraints based on process type (FR6)"""
        constraints = []
        task_ids = {task.id for task in tasks}
        
        # Find KYC tasks
        kyc_tasks = {
            t.id for t in tasks 
            if "kyc" in t.name.lower() or "identity" in t.name.lower() 
            or "verification" in t.name.lower()
        }
        
        # Find fraud check tasks
        fraud_tasks = {
            t.id for t in tasks 
            if "fraud" in t.name.lower() or "anti-fraud" in t.name.lower()
        }
        
        # Find approval tasks
        approval_tasks = {
            t.id for t in tasks 
            if "approval" in t.name.lower() or "approve" in t.name.lower()
        }
        
        # KYC is mandatory for most banking processes
        if kyc_tasks and process_type in [
            BankingProcessType.LOAN_APPROVAL,
            BankingProcessType.ACCOUNT_OPENING,
            BankingProcessType.CREDIT_CARD_APPLICATION,
            BankingProcessType.MORTGAGE_PROCESSING
        ]:
            constraints.append(ComplianceConstraint(
                id="constraint_kyc",
                name="KYC Mandatory",
                constraint_type=ConstraintType.KYC_REQUIRED,
                applies_to_tasks=kyc_tasks,
                is_mandatory=True,
                description="KYC verification is mandatory and cannot be skipped"
            ))
        
        # Anti-fraud checks are mandatory
        if fraud_tasks:
            constraints.append(ComplianceConstraint(
                id="constraint_fraud",
                name="Anti-Fraud Check Mandatory",
                constraint_type=ConstraintType.ANTI_FRAUD_REQUIRED,
                applies_to_tasks=fraud_tasks,
                is_mandatory=True,
                description="Anti-fraud checks are mandatory"
            ))
        
        # Dual approval for high-value processes
        if approval_tasks and process_type in [
            BankingProcessType.LOAN_APPROVAL,
            BankingProcessType.MORTGAGE_PROCESSING
        ]:
            constraints.append(ComplianceConstraint(
                id="constraint_dual_approval",
                name="Dual Approval Required",
                constraint_type=ConstraintType.DUAL_APPROVAL,
                applies_to_tasks=approval_tasks,
                is_mandatory=True,
                description="Dual approval required for high-value transactions"
            ))
        
        # Mark all constraint tasks as critical (cannot skip)
        # Create a copy of the list to avoid modifying while iterating
        critical_constraints = []
        for constraint in constraints:
            for task_id in constraint.applies_to_tasks:
                critical_constraints.append(ComplianceConstraint(
                    id=f"constraint_critical_{task_id}",
                    name=f"Critical Task: {task_id}",
                    constraint_type=ConstraintType.CANNOT_SKIP,
                    description="This task is critical and cannot be skipped",
                    applies_to_tasks={task_id},
                    is_mandatory=True
                ))
        
        constraints.extend(critical_constraints)
        return constraints
    
    def _identify_critical_tasks(
        self, 
        tasks: List[Task], 
        process_type: BankingProcessType
    ) -> Set[str]:
        """Identify critical tasks that cannot be removed (FR7)"""
        critical = set()
        
        # Keywords that indicate critical tasks
        critical_keywords = [
            "kyc", "verification", "approval", "compliance",
            "fraud", "anti-fraud", "credit check", "risk"
        ]
        
        for task in tasks:
            task_text = f"{task.name} {task.description}".lower()
            if any(keyword in task_text for keyword in critical_keywords):
                critical.add(task.id)
        
        return critical
    
    def analyze_process_stages(self, banking_process: BankingProcess) -> Dict[str, Any]:
        """Analyze and provide detailed stage information (FR2)"""
        stage_info = {
            "total_stages": len(banking_process.stages),
            "stages": [],
            "stage_sequence": []
        }
        
        for stage in banking_process.stages:
            stage_info["stages"].append({
                "name": stage.name,
                "value": stage.value,
                "description": f"Stage: {stage.value}"
            })
            stage_info["stage_sequence"].append(stage.value)
        
        return stage_info
    
    def get_dependency_graph(self, banking_process: BankingProcess) -> Dict[str, List[str]]:
        """Get dependency graph for visualization (FR3)"""
        graph = {}
        
        for dep in banking_process.task_dependencies:
            if dep.target_task_id not in graph:
                graph[dep.target_task_id] = []
            graph[dep.target_task_id].append(dep.source_task_id)
        
        return graph
