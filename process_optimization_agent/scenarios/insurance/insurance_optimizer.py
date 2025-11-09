"""
Insurance Process Optimizer
Handles optimization for all insurance process scenarios
"""

from typing import List, Dict, Set, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from ...Optimization.models import Process, Task, Resource, Schedule, ScheduleEntry
from .insurance_models import (
    InsuranceScenarioType, InsuranceMetrics, InsuranceProcess,
    InsuranceOptimizationResult, BottleneckAnalysis, TaskParallelizationOpportunity,
    OptimizationRecommendation, PayerType, UrgencyLevel, ComplexityLevel
)


def detect_user_involvement(process: Process) -> bool:
    """
    Detect if the insurance process involves direct user/patient/customer interaction.
    
    Returns:
        True if user is directly involved (use healthcare-style visualization)
        False if purely administrative (use manufacturing-style visualization)
    """
    # Keywords indicating user/patient is directly involved and waiting
    user_facing_indicators = [
        'patient', 'customer', 'client', 'approval', 'authorization', 
        'emergency', 'appeal', 'denial', 'notification', 'inquiry', 
        'complaint', 'waiting', 'appointment', 'consultation',
        'pre-auth', 'pre-authorization', 'eligibility check',
        'policyholder interaction', 'customer contact', 'member services'
    ]
    
    # Keywords indicating purely administrative/back-office work
    admin_only_indicators = [
        'batch', 'reconciliation', 'reporting', 'internal',
        'audit', 'compliance', 'record keeping', 'data entry',
        'filing', 'archiving', 'documentation', 'coding',
        'generation', 'processing', 'submission', 'verification',
        'premium collection', 'billing', 'invoice', 'payment processing',
        'account management', 'financial reporting', 'claims processing',
        'back office', 'administrative', 'internal process'
    ]
    
    user_score = 0
    admin_score = 0
    
    # Check process name and overview
    process_text = (process.name + ' ' + (process.description or '')).lower()
    
    for indicator in user_facing_indicators:
        if indicator in process_text:
            user_score += 2  # Process-level indicators are stronger
    
    for indicator in admin_only_indicators:
        if indicator in process_text:
            admin_score += 1
    
    # Check task names and descriptions
    for task in process.tasks:
        task_text = (task.name + ' ' + (task.description or '')).lower()
        
        for indicator in user_facing_indicators:
            if indicator in task_text:
                user_score += 1
        
        for indicator in admin_only_indicators:
            if indicator in task_text:
                admin_score += 1
    
    # Check user_involvement field in tasks (if present)
    admin_task_count = sum(1 for task in process.tasks if hasattr(task, 'user_involvement') and task.user_involvement == 'admin')
    if admin_task_count >= len(process.tasks) * 0.8:  # 80% or more are admin tasks
        admin_score += 10
    
    # Decision logic with enhanced robustness:
    # 1. Strong admin indicators mean it's back-office work
    if admin_score >= 8:
        return False  # Administrative only - use manufacturing visualization
    
    # 2. If admin score is significantly higher than user score, it's admin-only
    if admin_score > user_score * 2 and user_score < 5:
        return False  # Administrative only - use manufacturing visualization
    
    # 3. If user score is high and significantly higher than admin, user is involved
    if user_score >= 5 and user_score > admin_score * 1.2:
        return True  # User-facing - use healthcare visualization
    
    # 4. Default: if unclear, assume administrative (most insurance back-office is admin-only)
    # This is safer as it prevents empty visualizations
    return False


class InsuranceScenarioDetector:
    """Detects specific insurance scenario types"""
    
    def detect_scenario(self, process: Process) -> Tuple[InsuranceScenarioType, float, List[str]]:
        """
        Detect the specific insurance scenario type
        
        Returns:
            Tuple of (scenario_type, confidence, reasoning)
        """
        process_text = f"{process.name} {' '.join([t.name for t in process.tasks])}".lower()
        task_names = [t.name.lower() for t in process.tasks]
        
        scores = {}
        reasoning_map = {}
        
        # Scenario 1: Standard Billing
        standard_score, standard_reasons = self._check_standard_billing(process_text, task_names)
        scores[InsuranceScenarioType.STANDARD_BILLING] = standard_score
        reasoning_map[InsuranceScenarioType.STANDARD_BILLING] = standard_reasons
        
        # Scenario 2: Pre-Authorization
        preauth_score, preauth_reasons = self._check_pre_authorization(process_text, task_names)
        scores[InsuranceScenarioType.PRE_AUTHORIZATION] = preauth_score
        reasoning_map[InsuranceScenarioType.PRE_AUTHORIZATION] = preauth_reasons
        
        # Scenario 3: Emergency Care
        emergency_score, emergency_reasons = self._check_emergency_care(process_text, task_names)
        scores[InsuranceScenarioType.EMERGENCY_CARE] = emergency_score
        reasoning_map[InsuranceScenarioType.EMERGENCY_CARE] = emergency_reasons
        
        # Scenario 4: Denied Appeals
        appeal_score, appeal_reasons = self._check_denied_appeals(process_text, task_names)
        scores[InsuranceScenarioType.DENIED_APPEALS] = appeal_score
        reasoning_map[InsuranceScenarioType.DENIED_APPEALS] = appeal_reasons
        
        # Scenario 5: Multi-Payer
        multipayer_score, multipayer_reasons = self._check_multi_payer(process_text, task_names)
        scores[InsuranceScenarioType.MULTI_PAYER] = multipayer_score
        reasoning_map[InsuranceScenarioType.MULTI_PAYER] = multipayer_reasons
        
        # Scenario 6: Self-Pay
        selfpay_score, selfpay_reasons = self._check_self_pay(process_text, task_names)
        scores[InsuranceScenarioType.SELF_PAY] = selfpay_score
        reasoning_map[InsuranceScenarioType.SELF_PAY] = selfpay_reasons
        
        # Scenario 7: Workers' Comp
        wc_score, wc_reasons = self._check_workers_comp(process_text, task_names)
        scores[InsuranceScenarioType.WORKERS_COMP] = wc_score
        reasoning_map[InsuranceScenarioType.WORKERS_COMP] = wc_reasons
        
        # Scenario 8: Government Insurance
        gov_score, gov_reasons = self._check_government_insurance(process_text, task_names)
        scores[InsuranceScenarioType.GOVERNMENT_INSURANCE] = gov_score
        reasoning_map[InsuranceScenarioType.GOVERNMENT_INSURANCE] = gov_reasons
        
        # Scenario 9: Pharmacy/DME
        pharmacy_score, pharmacy_reasons = self._check_pharmacy_dme(process_text, task_names)
        scores[InsuranceScenarioType.PHARMACY_DME] = pharmacy_score
        reasoning_map[InsuranceScenarioType.PHARMACY_DME] = pharmacy_reasons
        
        # Scenario 10: Bundled Payments
        bundled_score, bundled_reasons = self._check_bundled_payments(process_text, task_names)
        scores[InsuranceScenarioType.BUNDLED_PAYMENTS] = bundled_score
        reasoning_map[InsuranceScenarioType.BUNDLED_PAYMENTS] = bundled_reasons
        
        # Scenario 11: Customer Service
        customer_score, customer_reasons = self._check_customer_service(process_text, task_names)
        scores[InsuranceScenarioType.CUSTOMER_SERVICE] = customer_score
        reasoning_map[InsuranceScenarioType.CUSTOMER_SERVICE] = customer_reasons
        
        # Scenario 12: Claims Processing
        claims_score, claims_reasons = self._check_claims_processing(process_text, task_names)
        scores[InsuranceScenarioType.CLAIMS_PROCESSING] = claims_score
        reasoning_map[InsuranceScenarioType.CLAIMS_PROCESSING] = claims_reasons
        
        # Determine best match
        best_scenario = max(scores, key=scores.get)
        best_score = scores[best_scenario]
        total_score = sum(scores.values())
        
        confidence = best_score / max(total_score, 1) if total_score > 0 else 0.0
        
        if confidence < 0.3 or best_score < 3:
            return InsuranceScenarioType.UNKNOWN, 0.0, ["No clear insurance scenario pattern detected"]
        
        return best_scenario, confidence, reasoning_map[best_scenario]
    
    def _check_standard_billing(self, process_text: str, task_names: List[str]) -> Tuple[float, List[str]]:
        """Check for standard billing scenario including premium collection"""
        score = 0.0
        reasons = []
        
        # Enhanced billing keywords including premium collection
        billing_terms = ['billing', 'invoice', 'charge', 'payment', 'claim submission', 'coding',
                        'premium', 'collection', 'receivable', 'account', 'financial report',
                        'reconciliation', 'statement', 'dunning', 'remittance']
        matches = sum(1 for term in billing_terms if term in process_text)
        
        if matches >= 3:
            score += 8
            reasons.append(f"Billing/Premium collection keywords detected ({matches} matches)")
        elif matches >= 2:
            score += 5
            reasons.append(f"Standard billing keywords detected ({matches} matches)")
        
        # Check for premium collection specific terms
        premium_terms = ['premium collection', 'premium billing', 'policy billing', 'premium payment']
        if any(term in process_text for term in premium_terms):
            score += 10
            reasons.append("Premium collection process detected")
        
        # Check for sequential billing tasks
        billing_sequence = ['invoice', 'generation', 'payment', 'processing', 'account', 'management',
                          'collection', 'reconciliation', 'reporting']
        sequence_matches = sum(1 for term in billing_sequence if any(term in task.lower() for task in task_names))
        
        if sequence_matches >= 4:
            score += 7
            reasons.append(f"Complete billing workflow detected ({sequence_matches} stages)")
        elif sequence_matches >= 2:
            score += 3
            reasons.append(f"Billing sequence detected ({sequence_matches} stages)")
        
        return score, reasons
    
    def _check_pre_authorization(self, process_text: str, task_names: List[str]) -> Tuple[float, List[str]]:
        """Check for pre-authorization scenario"""
        score = 0.0
        reasons = []
        
        preauth_terms = ['pre-auth', 'preauth', 'authorization', 'prior auth', 'approval']
        if any(term in process_text for term in preauth_terms):
            score += 10
            reasons.append("Pre-authorization keywords detected")
        
        if any('auth' in task for task in task_names):
            score += 5
            reasons.append("Authorization task found")
        
        return score, reasons
    
    def _check_emergency_care(self, process_text: str, task_names: List[str]) -> Tuple[float, List[str]]:
        """Check for emergency care scenario"""
        score = 0.0
        reasons = []
        
        emergency_terms = ['emergency', 'urgent', 'trauma', 'retroactive']
        if any(term in process_text for term in emergency_terms):
            score += 10
            reasons.append("Emergency care keywords detected")
        
        return score, reasons
    
    def _check_denied_appeals(self, process_text: str, task_names: List[str]) -> Tuple[float, List[str]]:
        """Check for denied appeals scenario"""
        score = 0.0
        reasons = []
        
        denial_terms = ['denial', 'denied', 'appeal', 'rejection', 'resubmit']
        matches = sum(1 for term in denial_terms if term in process_text)
        
        if matches >= 2:
            score += 10
            reasons.append(f"Denial/appeal keywords detected ({matches} terms)")
        
        if any('appeal' in task or 'denial' in task for task in task_names):
            score += 5
            reasons.append("Appeal/denial task found")
        
        return score, reasons
    
    def _check_multi_payer(self, process_text: str, task_names: List[str]) -> Tuple[float, List[str]]:
        """Check for multi-payer scenario"""
        score = 0.0
        reasons = []
        
        multipayer_terms = ['primary', 'secondary', 'tertiary', 'coordination of benefits', 'cob', 'multi-payer']
        if any(term in process_text for term in multipayer_terms):
            score += 10
            reasons.append("Multi-payer keywords detected")
        
        return score, reasons
    
    def _check_self_pay(self, process_text: str, task_names: List[str]) -> Tuple[float, List[str]]:
        """Check for self-pay scenario"""
        score = 0.0
        reasons = []
        
        selfpay_terms = ['self-pay', 'self pay', 'out-of-pocket', 'patient payment', 'cash payment']
        if any(term in process_text for term in selfpay_terms):
            score += 10
            reasons.append("Self-pay keywords detected")
        
        return score, reasons
    
    def _check_workers_comp(self, process_text: str, task_names: List[str]) -> Tuple[float, List[str]]:
        """Check for workers' compensation scenario"""
        score = 0.0
        reasons = []
        
        wc_terms = ['workers comp', 'workers compensation', 'wc', 'injury', 'employer notification']
        if any(term in process_text for term in wc_terms):
            score += 10
            reasons.append("Workers' compensation keywords detected")
        
        return score, reasons
    
    def _check_government_insurance(self, process_text: str, task_names: List[str]) -> Tuple[float, List[str]]:
        """Check for government insurance scenario"""
        score = 0.0
        reasons = []
        
        gov_terms = ['medicare', 'medicaid', 'cms', 'government insurance', 'cms-1500', 'ub-04']
        matches = sum(1 for term in gov_terms if term in process_text)
        
        if matches >= 1:
            score += 10
            reasons.append(f"Government insurance keywords detected ({matches} terms)")
        
        if any('compliance' in task for task in task_names):
            score += 3
            reasons.append("Compliance task found (common in government insurance)")
        
        return score, reasons
    
    def _check_pharmacy_dme(self, process_text: str, task_names: List[str]) -> Tuple[float, List[str]]:
        """Check for pharmacy/DME scenario"""
        score = 0.0
        reasons = []
        
        pharmacy_terms = ['pharmacy', 'prescription', 'formulary', 'dme', 'durable medical equipment', 'medication']
        if any(term in process_text for term in pharmacy_terms):
            score += 10
            reasons.append("Pharmacy/DME keywords detected")
        
        if any('formulary' in task or 'prescription' in task for task in task_names):
            score += 5
            reasons.append("Pharmacy-specific task found")
        
        return score, reasons
    
    def _check_bundled_payments(self, process_text: str, task_names: List[str]) -> Tuple[float, List[str]]:
        """Check for bundled payments scenario"""
        score = 0.0
        reasons = []
        
        bundled_terms = ['bundle', 'bundled', 'episode', 'risk adjustment', 'value-based']
        if any(term in process_text for term in bundled_terms):
            score += 10
            reasons.append("Bundled payment keywords detected")
        
        return score, reasons
    
    def _check_customer_service(self, process_text: str, task_names: List[str]) -> Tuple[float, List[str]]:
        """Check for customer service scenario"""
        score = 0.0
        reasons = []
        
        # Customer service sequence: Inquiry → Resolution → Assistance → Feedback → Training
        service_terms = ['inquiry', 'inquir', 'issue', 'resolution', 'resolv', 'assistance', 
                        'assist', 'feedback', 'training', 'support', 'customer', 'service']
        matches = sum(1 for term in service_terms if any(term in task.lower() for task in task_names))
        
        if matches >= 3:
            score += 10
            reasons.append(f"Customer service workflow detected ({matches} service-related tasks)")
        
        if 'customer service' in process_text or 'customer support' in process_text:
            score += 8
            reasons.append("Customer service process name detected")
        
        # Check for customer-facing indicators
        customer_terms = ['customer', 'policyholder', 'insured', 'client']
        if any(term in process_text for term in customer_terms):
            score += 5
            reasons.append("Customer-facing process indicators found")
        
        # Check for support activities
        support_terms = ['complaint', 'query', 'help', 'guidance', 'counseling']
        if any(term in process_text for term in support_terms):
            score += 3
            reasons.append("Support activity keywords detected")
        
        return score, reasons
    
    def _check_claims_processing(self, process_text: str, task_names: List[str]) -> Tuple[float, List[str]]:
        """Check for claims processing scenario"""
        score = 0.0
        reasons = []
        
        # Claims processing sequence: Submission → Verification → Assessment → Approval → Settlement
        claims_terms = ['claim', 'submission', 'review', 'verification', 'verify', 'assessment', 
                       'assess', 'approval', 'denial', 'settlement', 'payment', 'adjuster', 'examiner',
                       'investigation', 'evaluation', 'adjustment', 'loss', 'damage']
        matches = sum(1 for term in claims_terms if any(term in task.lower() for task in task_names))
        
        if matches >= 4:
            score += 12
            reasons.append(f"Strong claims processing workflow detected ({matches} claims-related tasks)")
        elif matches >= 3:
            score += 8
            reasons.append(f"Claims processing workflow detected ({matches} claims-related tasks)")
        
        if 'claims processing' in process_text or 'claim processing' in process_text or 'claims handling' in process_text:
            score += 10
            reasons.append("Claims processing process name detected")
        
        # Check for household/property/insurance claims indicators
        claims_types = ['household', 'property', 'auto', 'vehicle', 'home', 'renters', 'life', 'health',
                       'disability', 'accident', 'liability', 'casualty', 'marine', 'travel']
        type_matches = sum(1 for term in claims_types if term in process_text)
        if type_matches >= 1:
            score += 6
            reasons.append(f"Insurance claims type detected ({type_matches} types)")
        
        # Check for claims workflow stages
        workflow_stages = ['submission', 'verification', 'assessment', 'approval', 'settlement', 'payment']
        stage_matches = sum(1 for stage in workflow_stages if any(stage in task.lower() for task in task_names))
        if stage_matches >= 4:
            score += 9
            reasons.append(f"Complete claims workflow stages detected ({stage_matches}/6)")
        elif stage_matches >= 3:
            score += 6
            reasons.append(f"Standard claims workflow stages detected ({stage_matches}/6)")
        
        return score, reasons


class InsuranceProcessOptimizer:
    """
    Optimizer specialized for insurance processes
    Handles all 10 insurance scenario types
    """
    
    def __init__(self):
        self.scenario_detector = InsuranceScenarioDetector()
    
    def optimize(self, process: Process) -> InsuranceOptimizationResult:
        """
        Optimize insurance process based on detected scenario
        
        Returns:
            Complete optimization result with metrics and recommendations
        """
        # Detect specific scenario
        scenario_type, confidence, reasoning = self.scenario_detector.detect_scenario(process)
        
        # Calculate current state metrics
        current_metrics = self._calculate_current_metrics(process)
        
        # Apply scenario-specific optimization
        optimized_schedule, optimized_metrics = self._optimize_by_scenario(
            process, scenario_type
        )
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(process, current_metrics)
        
        # Find parallelization opportunities
        parallel_opps = self._find_parallelization_opportunities(process)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            scenario_type, bottlenecks, parallel_opps, current_metrics, optimized_metrics
        )
        
        # Create implementation plan
        implementation_phases = self._create_implementation_plan(recommendations)
        
        # Assess risks
        risks = self._assess_risks(scenario_type, recommendations)
        
        # Define success metrics
        success_metrics = self._define_success_metrics(scenario_type)
        
        # Detect user involvement
        # Customer service scenarios should use manufacturing visualization (admin-only)
        if scenario_type == InsuranceScenarioType.CUSTOMER_SERVICE:
            user_involved = False  # Use manufacturing-style visualization
        else:
            user_involved = detect_user_involvement(process)
        
        return InsuranceOptimizationResult(
            scenario_type=scenario_type,
            confidence=confidence,
            current_metrics=current_metrics,
            optimized_metrics=optimized_metrics,
            optimized_schedule=optimized_schedule,
            bottlenecks=bottlenecks,
            parallelization_opportunities=parallel_opps,
            recommendations=recommendations,
            implementation_phases=implementation_phases,
            risks=risks,
            success_metrics=success_metrics,
            user_involved=user_involved
        )
    
    def _calculate_current_metrics(self, process: Process) -> InsuranceMetrics:
        """Calculate current state metrics"""
        total_time = sum(task.duration_hours * 60 for task in process.tasks)  # Convert to minutes
        
        # Map tasks to resources based on job assignments
        task_resource_map = {}
        for task in process.tasks:
            # Try to find resource by matching task name keywords with resource skills/name
            task_lower = task.name.lower()
            matched_resource = None
            
            if 'bill' in task_lower:
                matched_resource = next((r for r in process.resources if 'billing' in r.name.lower()), None)
            elif 'insurance' in task_lower or 'verif' in task_lower or 'claim' in task_lower or 'reconcil' in task_lower:
                matched_resource = next((r for r in process.resources if 'insurance' in r.name.lower() or 'liaison' in r.name.lower()), None)
            elif 'record' in task_lower or 'account' in task_lower:
                matched_resource = next((r for r in process.resources if 'accountant' in r.name.lower()), None)
            
            if matched_resource:
                task_resource_map[task.id] = matched_resource
        
        # Calculate resource utilization and costs
        resource_utilization = {}
        resource_workload = {}
        total_cost = 0
        
        for task in process.tasks:
            resource = task_resource_map.get(task.id)
            if resource:
                # Calculate cost
                cost = task.duration_hours * resource.hourly_rate
                total_cost += cost
                
                # Track workload
                if resource.name not in resource_workload:
                    resource_workload[resource.name] = 0
                resource_workload[resource.name] += task.duration_hours * 60  # minutes
        
        # Calculate utilization percentages
        for resource_name, workload_minutes in resource_workload.items():
            utilization = (workload_minutes / total_time * 100) if total_time > 0 else 0
            resource_utilization[resource_name] = utilization
        
        return InsuranceMetrics(
            total_process_time=total_time,
            total_labor_cost=total_cost,
            cost_per_claim=total_cost,
            resource_utilization=resource_utilization,
            before_process_time=total_time,
            before_cost=total_cost
        )
    
    def _optimize_by_scenario(self, process: Process, scenario_type: InsuranceScenarioType) -> Tuple[Schedule, InsuranceMetrics]:
        """Apply scenario-specific optimization with fallback to generic"""
        
        # Try scenario-specific optimization
        schedule = None
        metrics = None
        
        try:
            if scenario_type == InsuranceScenarioType.STANDARD_BILLING:
                schedule, metrics = self._optimize_standard_billing(process)
            elif scenario_type == InsuranceScenarioType.PRE_AUTHORIZATION:
                schedule, metrics = self._optimize_pre_authorization(process)
            elif scenario_type == InsuranceScenarioType.EMERGENCY_CARE:
                schedule, metrics = self._optimize_emergency_care(process)
            elif scenario_type == InsuranceScenarioType.DENIED_APPEALS:
                schedule, metrics = self._optimize_denied_appeals(process)
            elif scenario_type == InsuranceScenarioType.MULTI_PAYER:
                schedule, metrics = self._optimize_multi_payer(process)
            elif scenario_type == InsuranceScenarioType.GOVERNMENT_INSURANCE:
                schedule, metrics = self._optimize_government_insurance(process)
            elif scenario_type == InsuranceScenarioType.CUSTOMER_SERVICE:
                schedule, metrics = self._optimize_customer_service(process)
            elif scenario_type == InsuranceScenarioType.CLAIMS_PROCESSING:
                schedule, metrics = self._optimize_claims_processing(process)
            else:
                # For unknown scenarios, use generic optimizer
                return self._optimize_generic_insurance_process(process)
            
            # If scenario-specific optimization produced empty schedule, fall back to generic
            if schedule and len(schedule.entries) == 0:
                print(f"   [WARNING] Scenario-specific optimization produced empty schedule, falling back to generic")
                return self._optimize_generic_insurance_process(process)
            
            return schedule, metrics
            
        except Exception as e:
            # If scenario-specific optimization fails, fall back to generic
            print(f"   [WARNING] Scenario-specific optimization failed: {str(e)}, falling back to generic")
            return self._optimize_generic_insurance_process(process)
    
    def _optimize_standard_billing(self, process: Process) -> Tuple[Schedule, InsuranceMetrics]:
        """
        Optimize standard billing scenario
        Strategy: Parallelize bill generation and verification
        """
        from ...Optimization.models import ScheduleEntry
        from datetime import datetime, timedelta
        
        schedule = Schedule(process_id=process.id)
        base_time = datetime.now()
        
        # Map tasks to resources
        task_resource_map = {}
        for task in process.tasks:
            task_lower = task.name.lower()
            matched_resource = None
            
            if 'bill' in task_lower:
                matched_resource = next((r for r in process.resources if 'billing' in r.name.lower()), None)
            elif 'insurance' in task_lower or 'verif' in task_lower or 'claim' in task_lower or 'reconcil' in task_lower:
                matched_resource = next((r for r in process.resources if 'insurance' in r.name.lower() or 'liaison' in r.name.lower()), None)
            elif 'record' in task_lower or 'account' in task_lower:
                matched_resource = next((r for r in process.resources if 'accountant' in r.name.lower()), None)
            
            if matched_resource:
                task_resource_map[task.id] = matched_resource
        
        # Identify tasks
        bill_tasks = [t for t in process.tasks if 'bill' in t.name.lower()]
        verify_tasks = [t for t in process.tasks if 'verif' in t.name.lower()]
        submit_tasks = [t for t in process.tasks if 'submit' in t.name.lower() or 'submission' in t.name.lower()]
        reconcile_tasks = [t for t in process.tasks if 'reconcil' in t.name.lower()]
        record_tasks = [t for t in process.tasks if 'record' in t.name.lower()]
        
        current_time = 0.0
        
        # Parallel Block 1: Bill generation and verification run simultaneously
        if bill_tasks and verify_tasks:
            # Bill generation starts at time 0
            bill_task = bill_tasks[0]
            bill_resource = task_resource_map.get(bill_task.id)
            if bill_resource:
                start_dt = base_time + timedelta(hours=current_time)
                end_dt = base_time + timedelta(hours=current_time + bill_task.duration_hours)
                schedule.add_entry(ScheduleEntry(
                    task_id=bill_task.id,
                    resource_id=bill_resource.id,
                    start_time=start_dt,
                    end_time=end_dt,
                    start_hour=current_time,
                    end_hour=current_time + bill_task.duration_hours
                ))
            
            # Verification also starts at time 0 (parallel)
            verify_task = verify_tasks[0]
            verify_resource = task_resource_map.get(verify_task.id)
            if verify_resource:
                start_dt = base_time + timedelta(hours=current_time)
                end_dt = base_time + timedelta(hours=current_time + verify_task.duration_hours)
                schedule.add_entry(ScheduleEntry(
                    task_id=verify_task.id,
                    resource_id=verify_resource.id,
                    start_time=start_dt,
                    end_time=end_dt,
                    start_hour=current_time,
                    end_hour=current_time + verify_task.duration_hours
                ))
            
            # Record keeping starts after bill generation
            if record_tasks:
                record_task = record_tasks[0]
                record_resource = task_resource_map.get(record_task.id)
                record_start = current_time + bill_task.duration_hours
                if record_resource:
                    start_dt = base_time + timedelta(hours=record_start)
                    end_dt = base_time + timedelta(hours=record_start + record_task.duration_hours)
                    schedule.add_entry(ScheduleEntry(
                        task_id=record_task.id,
                        resource_id=record_resource.id,
                        start_time=start_dt,
                        end_time=end_dt,
                        start_hour=record_start,
                        end_hour=record_start + record_task.duration_hours
                    ))
            
            parallel_time = max(bill_task.duration_hours, verify_task.duration_hours)
            current_time += parallel_time
        
        # Sequential: Submit after both bill and verification complete
        if submit_tasks:
            submit_task = submit_tasks[0]
            submit_resource = task_resource_map.get(submit_task.id)
            if not submit_resource:
                # If no resource mapped, try to find one based on task name
                if 'claim' in submit_task.name.lower() or 'submit' in submit_task.name.lower():
                    submit_resource = next((r for r in process.resources if 'insurance' in r.name.lower() or 'liaison' in r.name.lower()), None)
            
            if submit_resource:
                start_dt = base_time + timedelta(hours=current_time)
                end_dt = base_time + timedelta(hours=current_time + submit_task.duration_hours)
                schedule.add_entry(ScheduleEntry(
                    task_id=submit_task.id,
                    resource_id=submit_resource.id,
                    start_time=start_dt,
                    end_time=end_dt,
                    start_hour=current_time,
                    end_hour=current_time + submit_task.duration_hours
                ))
                current_time += submit_task.duration_hours
        
        # Sequential: Reconcile after submission
        if reconcile_tasks:
            reconcile_task = reconcile_tasks[0]
            reconcile_resource = task_resource_map.get(reconcile_task.id)
            if reconcile_resource:
                start_dt = base_time + timedelta(hours=current_time)
                end_dt = base_time + timedelta(hours=current_time + reconcile_task.duration_hours)
                schedule.add_entry(ScheduleEntry(
                    task_id=reconcile_task.id,
                    resource_id=reconcile_resource.id,
                    start_time=start_dt,
                    end_time=end_dt,
                    start_hour=current_time,
                    end_hour=current_time + reconcile_task.duration_hours
                ))
            current_time += reconcile_task.duration_hours
        
        # Apply skill-based load balancing if there's resource overload
        schedule = self._apply_skill_based_load_balancing(process, schedule, base_time)
        
        # Recalculate current_time after load balancing
        if schedule.entries:
            current_time = max(entry.end_hour for entry in schedule.entries)
        
        # Calculate optimized metrics
        original_time = sum(t.duration_hours * 60 for t in process.tasks)
        time_saved = (original_time - (current_time * 60))
        
        metrics = InsuranceMetrics(
            total_process_time=current_time * 60,  # Convert to minutes
            after_process_time=current_time * 60,
            time_savings_minutes=time_saved,
            time_savings_percent=(time_saved / original_time * 100) if original_time > 0 else 0
        )
        
        return schedule, metrics
    
    def _apply_skill_based_load_balancing(self, process: Process, schedule: Schedule, base_time) -> Schedule:
        """
        Apply skill-based load balancing to redistribute tasks from overloaded resources
        to underutilized resources based on skill compatibility.
        """
        from .models import ScheduleEntry
        from datetime import timedelta
        
        # Calculate resource workload
        resource_workload = {}
        resource_tasks = {}
        
        for entry in schedule.entries:
            resource_id = entry.resource_id
            duration = entry.end_hour - entry.start_hour
            
            if resource_id not in resource_workload:
                resource_workload[resource_id] = 0
                resource_tasks[resource_id] = []
            
            resource_workload[resource_id] += duration
            resource_tasks[resource_id].append(entry)
        
        # Find overloaded resources (>60% utilization threshold)
        max_hours = 0.75  # 45 minutes for this process
        overload_threshold = 0.6  # 60% utilization
        
        overloaded_resources = []
        underutilized_resources = []
        
        for resource in process.resources:
            workload = resource_workload.get(resource.id, 0)
            utilization = workload / max_hours if max_hours > 0 else 0
            
            if utilization > overload_threshold:
                overloaded_resources.append((resource, workload, utilization))
            elif utilization < 0.4:  # Less than 40% utilized
                underutilized_resources.append((resource, workload, utilization))
        
        # If no overload, return original schedule
        if not overloaded_resources or not underutilized_resources:
            return schedule
        
        # Check skill compatibility and redistribute
        new_entries = []
        redistributed = False
        
        for entry in schedule.entries:
            task = process.get_task_by_id(entry.task_id)
            current_resource = process.get_resource_by_id(entry.resource_id)
            
            # Check if this task is from an overloaded resource
            is_overloaded = any(r.id == current_resource.id for r, _, _ in overloaded_resources)
            
            if is_overloaded and task:
                # Check if task can be redistributed based on skill compatibility
                task_name_lower = task.name.lower()
                
                # Reconciliation tasks can be moved to accountants (skill-compatible)
                if 'reconcil' in task_name_lower:
                    # Find accountant resource
                    accountant = next((r for r, _, _ in underutilized_resources 
                                     if 'accountant' in r.name.lower()), None)
                    
                    if accountant:
                        # Redistribute to accountant
                        # Find the latest end time for the accountant
                        accountant_end_time = max(
                            (e.end_hour for e in new_entries if e.resource_id == accountant.id),
                            default=0
                        )
                        
                        new_start = accountant_end_time
                        new_end = new_start + (entry.end_hour - entry.start_hour)
                        
                        new_entry = ScheduleEntry(
                            task_id=entry.task_id,
                            resource_id=accountant.id,
                            start_time=base_time + timedelta(hours=new_start),
                            end_time=base_time + timedelta(hours=new_end),
                            start_hour=new_start,
                            end_hour=new_end
                        )
                        new_entries.append(new_entry)
                        redistributed = True
                        continue
            
            # Keep original entry if not redistributed
            new_entries.append(entry)
        
        # Update schedule with new entries if redistribution occurred
        if redistributed:
            schedule.entries = new_entries
        
        return schedule
    
    def _optimize_pre_authorization(self, process: Process) -> Tuple[Schedule, InsuranceMetrics]:
        """Optimize pre-authorization scenario"""
        # Similar structure but with pre-auth specific logic
        return self._optimize_standard_billing(process)
    
    def _optimize_emergency_care(self, process: Process) -> Tuple[Schedule, InsuranceMetrics]:
        """Optimize emergency care scenario"""
        return self._optimize_standard_billing(process)
    
    def _optimize_denied_appeals(self, process: Process) -> Tuple[Schedule, InsuranceMetrics]:
        """Optimize denied appeals scenario"""
        return self._optimize_standard_billing(process)
    
    def _optimize_multi_payer(self, process: Process) -> Tuple[Schedule, InsuranceMetrics]:
        """Optimize multi-payer scenario"""
        return self._optimize_standard_billing(process)
    
    def _optimize_government_insurance(self, process: Process) -> Tuple[Schedule, InsuranceMetrics]:
        """Optimize government insurance scenario"""
        return self._optimize_standard_billing(process)
    
    def _identify_bottlenecks(self, process: Process, metrics: InsuranceMetrics) -> List[BottleneckAnalysis]:
        """Identify resource bottlenecks"""
        bottlenecks = []
        
        # Map tasks to resources
        task_resource_map = {}
        for task in process.tasks:
            task_lower = task.name.lower()
            matched_resource = None
            
            if 'bill' in task_lower:
                matched_resource = next((r for r in process.resources if 'billing' in r.name.lower()), None)
            elif 'insurance' in task_lower or 'verif' in task_lower or 'claim' in task_lower or 'reconcil' in task_lower:
                matched_resource = next((r for r in process.resources if 'insurance' in r.name.lower() or 'liaison' in r.name.lower()), None)
            elif 'record' in task_lower or 'account' in task_lower:
                matched_resource = next((r for r in process.resources if 'accountant' in r.name.lower()), None)
            
            if matched_resource:
                task_resource_map[task.id] = matched_resource
        
        for resource_name, utilization in metrics.resource_utilization.items():
            if utilization > 70:  # Bottleneck threshold
                # Find the resource
                resource = next((r for r in process.resources if r.name == resource_name), None)
                if resource:
                    # Find tasks assigned to this resource
                    assigned_tasks = [t.name for t in process.tasks 
                                    if task_resource_map.get(t.id) == resource]
                    
                    workload = sum(t.duration_hours * 60 for t in process.tasks 
                                 if task_resource_map.get(t.id) == resource)
                    
                    impact = "Critical" if utilization > 85 else "High" if utilization > 75 else "Medium"
                    
                    bottlenecks.append(BottleneckAnalysis(
                        resource_name=resource_name,
                        resource_id=resource.id,
                        total_workload_minutes=workload,
                        utilization_percent=utilization,
                        tasks_assigned=assigned_tasks,
                        impact_on_throughput=impact,
                        recommendations=[
                            f"Consider adding another {resource_name}",
                            f"Cross-train other staff to handle {resource_name} tasks",
                            "Implement automation for routine tasks"
                        ]
                    ))
        
        return bottlenecks
    
    def _find_parallelization_opportunities(self, process: Process) -> List[TaskParallelizationOpportunity]:
        """Find tasks that can be parallelized"""
        opportunities = []
        
        # Find tasks with no dependencies
        independent_tasks = [t for t in process.tasks if not t.dependencies]
        
        if len(independent_tasks) >= 2:
            time_saved = sum(t.duration_hours * 60 for t in independent_tasks[1:])
            opportunities.append(TaskParallelizationOpportunity(
                task_group=[t.id for t in independent_tasks],
                time_saved=time_saved,
                dependencies_satisfied=True,
                resource_availability=True,
                recommendation=f"Run {len(independent_tasks)} independent tasks in parallel to save {time_saved:.1f} minutes"
            ))
        
        return opportunities
    
    def _generate_recommendations(
        self, 
        scenario_type: InsuranceScenarioType,
        bottlenecks: List[BottleneckAnalysis],
        parallel_opps: List[TaskParallelizationOpportunity],
        current_metrics: InsuranceMetrics,
        optimized_metrics: InsuranceMetrics
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Immediate: Implement parallelization
        if parallel_opps:
            recommendations.append(OptimizationRecommendation(
                priority="IMMEDIATE",
                category="Process",
                title="Implement Parallel Processing",
                description="Run bill generation and insurance verification simultaneously",
                expected_impact=f"Reduce process time by {optimized_metrics.time_savings_percent:.1f}%",
                implementation_cost=0.0,
                roi_months=0.0,
                risk_level="Low",
                mitigation_strategies=["Test with pilot claims first", "Train staff on new workflow"]
            ))
        
        # Short-term: Address bottlenecks
        for bottleneck in bottlenecks:
            recommendations.append(OptimizationRecommendation(
                priority="SHORT_TERM",
                category="Resource",
                title=f"Address {bottleneck.resource_name} Bottleneck",
                description=f"Resource utilization at {bottleneck.utilization_percent:.1f}% - consider staffing adjustments",
                expected_impact=f"Increase capacity by 20-30%",
                implementation_cost=5000.0,
                roi_months=3.0,
                risk_level="Medium",
                mitigation_strategies=bottleneck.recommendations
            ))
        
        return recommendations
    
    def _create_implementation_plan(self, recommendations: List[OptimizationRecommendation]) -> List[Dict[str, Any]]:
        """Create phased implementation plan"""
        phases = [
            {
                "phase": "Phase 1: Preparation (Weeks 1-2)",
                "activities": [
                    "Stakeholder kickoff meeting",
                    "Staff training on new workflow",
                    "System readiness check",
                    "Process documentation update"
                ]
            },
            {
                "phase": "Phase 2: Pilot (Weeks 3-4)",
                "activities": [
                    "Run optimized process on 20 pilot claims",
                    "Daily monitoring and issue logging",
                    "Staff feedback collection",
                    "Performance metrics tracking"
                ]
            },
            {
                "phase": "Phase 3: Rollout (Weeks 5-8)",
                "activities": [
                    "Train all remaining staff",
                    "Gradual transition to optimized process",
                    "Intensive support and monitoring",
                    "Celebrate quick wins"
                ]
            },
            {
                "phase": "Phase 4: Stabilization (Weeks 9-12)",
                "activities": [
                    "Weekly performance reviews",
                    "Continuous improvement sessions",
                    "Document lessons learned",
                    "Plan next optimization phase"
                ]
            }
        ]
        return phases
    
    def _assess_risks(self, scenario_type: InsuranceScenarioType, recommendations: List[OptimizationRecommendation]) -> List[Dict[str, Any]]:
        """Assess implementation risks"""
        risks = [
            {
                "risk": "Quality Degradation",
                "probability": "Medium",
                "impact": "High",
                "mitigation": "Implement quality checkpoints, random audits of 10% of claims"
            },
            {
                "risk": "Staff Resistance",
                "probability": "High",
                "impact": "Medium",
                "mitigation": "Involve staff in design, provide training, gradual rollout"
            },
            {
                "risk": "Technology Failures",
                "probability": "Low",
                "impact": "High",
                "mitigation": "Ensure system reliability, implement fallback procedures"
            }
        ]
        return risks
    
    def _optimize_customer_service(self, process: Process) -> Tuple[Schedule, InsuranceMetrics]:
        """
        Optimize customer service scenario
        Strategy: Sequential workflow with load balancing for high-volume tasks
        Prioritize inquiry handling and issue resolution
        """
        from ...Optimization.optimizers import ProcessOptimizer
        from ...Optimization.models import ScheduleEntry
        from datetime import datetime, timedelta
        
        # Use standard optimizer with sequential execution
        optimizer = ProcessOptimizer()
        schedule = optimizer.optimize(process)
        
        # Calculate metrics
        if schedule.entries:
            total_time = max([entry.end_hour for entry in schedule.entries])
            total_cost = sum([
                (entry.end_hour - entry.start_hour) * process.get_resource_by_id(entry.resource_id).hourly_rate
                for entry in schedule.entries
            ])
        else:
            total_time = 0
            total_cost = 0
        
        # Calculate before metrics
        before_time = sum([t.duration_hours * 60 for t in process.tasks])
        before_cost = sum([
            t.duration_hours * next((r.hourly_rate for r in process.resources 
                                    if any(s.name in [rs.name for rs in t.required_skills] 
                                          for s in r.skills)), 50)
            for t in process.tasks
        ])
        
        metrics = InsuranceMetrics(
            total_process_time=total_time * 60,  # Convert to minutes
            total_labor_cost=total_cost,
            before_process_time=before_time,
            after_process_time=total_time * 60,
            before_cost=before_cost,
            after_cost=total_cost
        )
        
        return schedule, metrics
    
    def _optimize_claims_processing(self, process: Process) -> Tuple[Schedule, InsuranceMetrics]:
        """
        Optimize claims processing scenario
        Strategy: Sequential workflow respecting dependencies
        Submission → Verification → Assessment → Approval → Settlement
        """
        from ...Optimization.optimizers import ProcessOptimizer
        from ...Optimization.models import ScheduleEntry
        from datetime import datetime, timedelta
        
        # Use standard optimizer with sequential execution respecting dependencies
        optimizer = ProcessOptimizer()
        schedule = optimizer.optimize(process)
        
        # Calculate metrics
        if schedule.entries:
            total_time = max([entry.end_hour for entry in schedule.entries])
            total_cost = sum([
                (entry.end_hour - entry.start_hour) * process.get_resource_by_id(entry.resource_id).hourly_rate
                for entry in schedule.entries
            ])
        else:
            total_time = 0
            total_cost = 0
        
        # Calculate before metrics
        before_time = sum([t.duration_hours * 60 for t in process.tasks])
        
        # Calculate before cost by matching tasks to resources
        before_cost = 0
        for task in process.tasks:
            # Find matching resource for this task
            matched_rate = 50  # Default rate
            for resource in process.resources:
                # Check if any task skill matches any resource skill
                if any(task_skill.name == res_skill.name 
                      for task_skill in task.required_skills 
                      for res_skill in resource.skills):
                    matched_rate = resource.hourly_rate
                    break
            before_cost += task.duration_hours * matched_rate
        
        metrics = InsuranceMetrics(
            total_process_time=total_time * 60,  # Convert to minutes
            total_labor_cost=total_cost,
            before_process_time=before_time,
            after_process_time=total_time * 60,
            before_cost=before_cost,
            after_cost=total_cost
        )
        
        return schedule, metrics
    
    def _optimize_generic_insurance_process(self, process: Process) -> Tuple[Schedule, InsuranceMetrics]:
        """
        Generic optimization for insurance processes that don't match specific scenarios
        Uses standard process optimizer with sequential task execution
        """
        from ...Optimization.optimizers import ProcessOptimizer
        from datetime import datetime
        
        # Use standard optimizer for generic insurance processes
        optimizer = ProcessOptimizer()
        schedule = optimizer.optimize(process)
        
        # Calculate metrics from schedule
        if schedule.entries:
            total_time = max([entry.end_hour for entry in schedule.entries])
            total_cost = sum([
                (entry.end_hour - entry.start_hour) * process.get_resource_by_id(entry.resource_id).hourly_rate
                for entry in schedule.entries
            ])
        else:
            total_time = 0
            total_cost = 0
        
        # Calculate before cost with proper skill matching
        before_cost = 0
        for task in process.tasks:
            matched_rate = 50
            for resource in process.resources:
                if any(task_skill.name == res_skill.name 
                       for task_skill in task.required_skills 
                       for res_skill in resource.skills):
                    matched_rate = resource.hourly_rate
                    break
            before_cost += task.duration_hours * matched_rate
        
        metrics = InsuranceMetrics(
            total_process_time=total_time * 60,  # Convert to minutes
            total_labor_cost=total_cost,
            before_process_time=sum([t.duration_hours * 60 for t in process.tasks]),
            after_process_time=total_time * 60,
            before_cost=before_cost,
            after_cost=total_cost
        )
        
        return schedule, metrics
    
    def _define_success_metrics(self, scenario_type: InsuranceScenarioType) -> List[str]:
        """Define success metrics for tracking"""
        return [
            "Process time (target: 25% reduction)",
            "Cost per claim (maintain or reduce)",
            "Error rate (target: <2%)",
            "Staff satisfaction (target: >80%)",
            "Claim approval rate (target: >95%)",
            "Daily throughput (target: 30% increase)"
        ]
