"""
Process Intelligence Module - Automatic process type detection and classification
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
from .models import Process, Task, Resource

class ProcessType(Enum):
    """Process type classifications"""
    HEALTHCARE = "healthcare"
    MANUFACTURING = "manufacturing"
    BANKING = "banking"
    ACADEMIC = "academic"
    UNKNOWN = "unknown"

class OptimizationStrategy(Enum):
    """Optimization strategy types"""
    SEQUENTIAL_USER = "sequential_user"  # Single user journey (healthcare)
    PARALLEL_PRODUCTION = "parallel_production"  # Manufacturing/production
    CONDITIONAL_APPROVAL = "conditional_approval"  # Banking/approval workflows
    MIXED_ACADEMIC = "mixed_academic"  # Academic processes
    BALANCED = "balanced"  # Default strategy

@dataclass
class ProcessClassification:
    """Result of process classification"""
    process_type: ProcessType
    confidence: float
    optimization_strategy: OptimizationStrategy
    characteristics: Dict[str, Any]
    reasoning: List[str]

class ProcessIntelligence:
    """Intelligent process type detection and classification"""
    
    def __init__(self):
        self.initialize_patterns()
    
    def initialize_patterns(self):
        """Initialize detection patterns for different process types"""
        
        self.patterns = {
            ProcessType.HEALTHCARE: {
                'keywords': [
                    # Core healthcare terms
                    'patient', 'doctor', 'nurse', 'physician', 'medical', 'health', 'healthcare',
                    'clinical', 'clinic', 'hospital', 'medicine', 'nursing',
                    
                    # Medical procedures and services
                    'treatment', 'consultation', 'examination', 'diagnosis', 'therapy',
                    'surgery', 'operation', 'procedure', 'intervention', 'care',
                    'screening', 'checkup', 'followup', 'appointment',
                    
                    # Medical specialties
                    'cardiology', 'oncology', 'pediatric', 'radiology', 'pathology',
                    'neurology', 'orthopedic', 'dermatology', 'psychiatry',
                    
                    # Healthcare facilities and departments
                    'emergency', 'icu', 'intensive care', 'outpatient', 'inpatient',
                    'ward', 'operating room', 'laboratory', 'pharmacy', 'radiology',
                    
                    # Medical documentation and processes
                    'prescription', 'medication', 'drug', 'vaccine', 'immunization',
                    'vital signs', 'blood pressure', 'temperature', 'pulse',
                    'symptom', 'complaint', 'illness', 'disease', 'condition',
                    'injury', 'wound', 'pain', 'fever',
                    
                    # Healthcare administration
                    'admission', 'discharge', 'transfer', 'referral', 'triage',
                    'medical records', 'health records', 'chart', 'ehr', 'emr',
                    'insurance', 'billing', 'claim', 'copay'
                ],
                'task_patterns': [
                    # Patient intake and registration
                    'patient registration', 'check-in', 'admission', 'intake',
                    'patient greeting', 'form distribution', 'identity verification',
                    'insurance verification', 'medical history',
                    
                    # Medical assessments
                    'examination', 'assessment', 'evaluation', 'diagnosis',
                    'vital signs', 'physical exam', 'initial assessment',
                    'triage', 'screening', 'consultation',
                    
                    # Medical procedures
                    'treatment', 'therapy', 'procedure', 'surgery', 'operation',
                    'intervention', 'medication administration', 'injection',
                    'blood draw', 'specimen collection', 'imaging', 'x-ray', 'scan',
                    
                    # Documentation and follow-up
                    'prescription', 'discharge', 'follow-up', 'referral',
                    'medical records', 'documentation', 'charting', 'notes',
                    'test results', 'lab results', 'report',
                    
                    # Administrative tasks
                    'scheduling', 'appointment', 'billing', 'payment',
                    'insurance processing', 'claim submission', 'authorization'
                ],
                'resource_patterns': [
                    # Medical professionals
                    'physician', 'doctor', 'nurse', 'surgeon', 'specialist',
                    'practitioner', 'clinician', 'medical officer', 'attending',
                    
                    # Nursing staff
                    'registered nurse', 'rn', 'lpn', 'nurse practitioner',
                    'nursing assistant', 'cna', 'charge nurse',
                    
                    # Allied health professionals
                    'pharmacist', 'therapist', 'technician', 'radiologist',
                    'lab technician', 'medical technician', 'phlebotomist',
                    'respiratory therapist', 'physical therapist',
                    
                    # Administrative and support staff
                    'receptionist', 'medical receptionist', 'scheduler',
                    'medical records', 'billing specialist', 'coder',
                    'registration clerk', 'admissions', 'coordinator',
                    'medical assistant', 'health assistant', 'care coordinator'
                ],
                'characteristics': {
                    'user_centric': True,
                    'sequential_flow': True,
                    'parallelism_level': 'low',
                    'human_interaction': 'high'
                }
            },
            
            ProcessType.MANUFACTURING: {
                'keywords': [
                    # Core manufacturing terms
                    'manufacturing', 'production', 'factory', 'plant', 'facility',
                    'assembly', 'fabrication', 'processing', 'machining',
                    
                    # Production processes
                    'build', 'make', 'create', 'construct', 'develop', 'produce',
                    'manufacture', 'assemble', 'fabricate', 'process',
                    
                    # Quality and testing
                    'quality', 'inspection', 'testing', 'qa', 'qc', 'quality control',
                    'quality assurance', 'verification', 'validation', 'audit',
                    
                    # Materials and components
                    'material', 'component', 'part', 'product', 'item', 'unit',
                    'raw material', 'finished goods', 'work in progress', 'wip',
                    'inventory', 'stock', 'supply', 'batch', 'lot',
                    
                    # Equipment and machinery
                    'machine', 'equipment', 'tool', 'apparatus', 'device',
                    'machinery', 'automation', 'robot', 'cnc', 'lathe',
                    
                    # Logistics and storage
                    'warehouse', 'storage', 'packaging', 'shipping', 'logistics',
                    'distribution', 'dispatch', 'delivery', 'transport',
                    
                    # Software development (manufacturing of software)
                    'development', 'coding', 'programming', 'software', 'application',
                    'platform', 'system', 'module', 'feature', 'functionality',
                    'api', 'backend', 'frontend', 'database', 'interface',
                    'deployment', 'release', 'version', 'build', 'compile',
                    
                    # Project and process management
                    'workflow', 'pipeline', 'process', 'procedure', 'operation',
                    'cycle', 'stage', 'phase', 'step', 'milestone'
                ],
                'task_patterns': [
                    # Physical manufacturing tasks
                    'cutting', 'welding', 'drilling', 'milling', 'grinding',
                    'assembly', 'fabrication', 'machining', 'forming', 'shaping',
                    'molding', 'casting', 'forging', 'stamping', 'bending',
                    'painting', 'coating', 'finishing', 'polishing', 'cleaning',
                    
                    # Quality and inspection
                    'testing', 'inspection', 'quality check', 'qa', 'qc',
                    'verification', 'validation', 'audit', 'review',
                    'measurement', 'calibration', 'analysis',
                    
                    # Packaging and logistics
                    'packaging', 'packing', 'labeling', 'sorting', 'organizing',
                    'loading', 'unloading', 'shipping', 'dispatch',
                    
                    # Software development tasks
                    'design', 'development', 'coding', 'programming', 'implementation',
                    'database design', 'api development', 'frontend development',
                    'backend development', 'ui design', 'ux design',
                    'authentication', 'authorization', 'integration',
                    'deployment', 'configuration', 'setup', 'installation',
                    'testing', 'debugging', 'troubleshooting', 'optimization',
                    
                    # Process management
                    'planning', 'scheduling', 'coordination', 'monitoring',
                    'tracking', 'reporting', 'documentation'
                ],
                'resource_patterns': [
                    # Manufacturing personnel
                    'operator', 'machinist', 'assembler', 'fabricator',
                    'technician', 'mechanic', 'welder', 'painter',
                    
                    # Quality and inspection
                    'inspector', 'quality engineer', 'qa engineer', 'qc inspector',
                    'quality analyst', 'tester', 'auditor',
                    
                    # Engineering and technical
                    'engineer', 'mechanical engineer', 'electrical engineer',
                    'industrial engineer', 'process engineer', 'design engineer',
                    'manufacturing engineer', 'production engineer',
                    
                    # Software development
                    'developer', 'programmer', 'software engineer', 'coder',
                    'frontend developer', 'backend developer', 'fullstack developer',
                    'full stack developer', 'web developer', 'mobile developer',
                    'database administrator', 'dba', 'devops engineer',
                    'ui designer', 'ux designer', 'architect', 'tech lead',
                    
                    # Management and supervision
                    'supervisor', 'manager', 'team lead', 'coordinator',
                    'production manager', 'plant manager', 'operations manager',
                    'project manager', 'scrum master', 'product owner',
                    
                    # Specialized roles
                    'specialist', 'analyst', 'consultant', 'expert',
                    'payment specialist', 'integration specialist', 'security specialist'
                ],
                'characteristics': {
                    'user_centric': False,
                    'sequential_flow': False,
                    'parallelism_level': 'high',
                    'human_interaction': 'low'
                }
            },
            
            ProcessType.BANKING: {
                'keywords': [
                    'loan', 'credit', 'bank', 'finance', 'financial',
                    'transaction', 'account opening', 'deposit', 'withdrawal',
                    'compliance', 'risk assessment', 'kyc', 'fraud detection',
                    'disbursement', 'mortgage', 'lending', 'credit score',
                    'income verification', 'debt', 'interest rate'
                ],
                'task_patterns': [
                    'loan application', 'credit check', 'disbursement',
                    'credit verification', 'income verification',
                    'fraud check', 'anti-fraud', 'compliance check',
                    'risk assessment', 'credit score check',
                    'kyc verification', 'account opening', 'account setup',
                    'loan approval', 'credit approval', 'financial review'
                ],
                'resource_patterns': [
                    'loan officer', 'credit analyst', 'bank manager',
                    'compliance officer', 'underwriter', 'credit reviewer',
                    'kyc analyst', 'fraud analyst', 'risk analyst',
                    'teller', 'banker', 'account manager', 'risk manager',
                    'financial advisor', 'credit specialist'
                ],
                'characteristics': {
                    'user_centric': True,
                    'sequential_flow': False,
                    'parallelism_level': 'medium',
                    'human_interaction': 'medium'
                }
            },
            
            ProcessType.ACADEMIC: {
                'keywords': [
                    'student', 'course', 'registration', 'enrollment',
                    'grade', 'transcript', 'academic', 'education',
                    'class', 'semester', 'degree', 'admission',
                    'faculty', 'department', 'prerequisite'
                ],
                'task_patterns': [
                    'application', 'prerequisite check', 'approval',
                    'enrollment', 'registration', 'evaluation',
                    'submission', 'review', 'notification'
                ],
                'resource_patterns': [
                    'advisor', 'registrar', 'teacher', 'admin',
                    'faculty', 'coordinator', 'counselor', 'staff'
                ],
                'characteristics': {
                    'user_centric': True,
                    'sequential_flow': False,
                    'parallelism_level': 'medium',
                    'human_interaction': 'high'
                }
            }
        }
    
    def detect_process_type(self, process: Process) -> ProcessClassification:
        """
        Automatically detect the type of process based on various indicators
        
        Args:
            process: The process to classify
            
        Returns:
            ProcessClassification with type, confidence, and strategy
        """
        # CRITICAL RULE: If "patient" is mentioned anywhere, it's ALWAYS healthcare
        process_text = f"{process.name} {' '.join([t.name for t in process.tasks])} {' '.join([r.name for r in process.resources])}".lower()
        if 'patient' in process_text:
            return ProcessClassification(
                process_type=ProcessType.HEALTHCARE,
                confidence=0.99,  # Very high confidence
                optimization_strategy=OptimizationStrategy.SEQUENTIAL_USER,
                characteristics=self.patterns[ProcessType.HEALTHCARE]['characteristics'],
                reasoning=[
                    "CRITICAL RULE: 'Patient' keyword detected - automatically classified as HEALTHCARE",
                    "Patient-centric processes are always healthcare by definition"
                ]
            )
        
        scores = {}
        reasoning = {}
        
        # ONLY evaluate Healthcare and Manufacturing (Banking and Academic disabled)
        enabled_types = [ProcessType.HEALTHCARE, ProcessType.MANUFACTURING]
        
        for process_type in enabled_types:
            patterns = self.patterns[process_type]
            score = 0
            reasons = []
            
            # 1. Analyze process name
            name_score = self._analyze_text(
                process.name.lower(), 
                patterns['keywords']
            )
            if name_score > 0:
                score += name_score * 3  # Name is heavily weighted
                reasons.append(f"Process name matches {process_type.value} keywords")
            
            # 2. Analyze task names and descriptions
            task_text = ' '.join([
                f"{task.name} {task.description}".lower() 
                for task in process.tasks
            ])
            task_keyword_score = self._analyze_text(task_text, patterns['keywords'])
            task_pattern_score = self._analyze_text(task_text, patterns['task_patterns'])
            
            # Also analyze just task names for pattern matching
            task_names_text = ' '.join([task.name.lower() for task in process.tasks])
            task_name_pattern_score = self._analyze_text(task_names_text, patterns['task_patterns'])
            
            if task_keyword_score > 0:
                score += task_keyword_score * 2
                reasons.append(f"Task names match {process_type.value} domain")
            
            if task_pattern_score > 0:
                score += task_pattern_score * 2
                reasons.append(f"Task patterns match {process_type.value} workflows")
            
            if task_name_pattern_score > 0:
                score += task_name_pattern_score * 3  # Higher weight for task name patterns
                reasons.append(f"Task names contain {process_type.value} specific terms")
            
            # 3. Analyze resource types
            resource_text = ' '.join([
                f"{res.name} {' '.join([s.name for s in res.skills])}".lower()
                for res in process.resources
            ])
            resource_score = self._analyze_text(resource_text, patterns['resource_patterns'])
            
            if resource_score > 0:
                score += resource_score * 2
                reasons.append(f"Resources match {process_type.value} roles")
            
            # 4. Analyze task dependencies structure
            dependency_score = self._analyze_dependency_structure(process, process_type)
            if dependency_score > 0:
                score += dependency_score
                reasons.append(f"Dependency structure matches {process_type.value}")
            
            # 5. Check for specific indicators
            if process_type == ProcessType.HEALTHCARE:
                # Strong healthcare indicators
                healthcare_count = 0
                process_name_lower = process.name.lower()
                
                # Very strong indicator: "patient" in process name
                if 'patient' in process_name_lower:
                    healthcare_count += 2  # Count as 2 indicators
                    score += 20  # Very high boost
                    reasons.append("'Patient' keyword in process name (strong healthcare indicator)")
                
                # Strong healthcare process names
                if 'registration' in process_name_lower and 'patient' in process_name_lower:
                    score += 15
                    healthcare_count += 1
                    reasons.append("Patient registration process detected")
                
                if 'consultation' in process_name_lower or 'clinic' in process_name_lower:
                    healthcare_count += 1
                    score += 8
                
                # Healthcare resources
                if any('doctor' in r.name.lower() or 'physician' in r.name.lower() or 'nurse' in r.name.lower() 
                       for r in process.resources):
                    healthcare_count += 1
                    score += 10
                    reasons.append("Healthcare professionals detected in resources")
                
                # Healthcare-specific resources
                if any('receptionist' in r.name.lower() or 'medical records' in r.name.lower() 
                       for r in process.resources):
                    healthcare_count += 1
                    score += 8
                
                # Healthcare tasks
                if any('medical' in t.name.lower() or 'diagnosis' in t.name.lower() or 'examination' in t.name.lower()
                       for t in process.tasks):
                    healthcare_count += 1
                    score += 8
                
                if healthcare_count >= 3:
                    score += 20  # Strong boost for multiple healthcare indicators
                    reasons.append("Multiple strong healthcare indicators found")
                
                # Healthcare processes often have a single entity flowing through
                if self._has_sequential_flow(process):
                    score += 5
                    reasons.append("Sequential flow pattern detected")
            
            elif process_type == ProcessType.MANUFACTURING:
                # Manufacturing often has parallel tasks
                if self._has_high_parallelism(process):
                    score += 5
                    reasons.append("High parallelism potential detected")
            
            elif process_type == ProcessType.BANKING:
                # Strong banking indicators
                banking_count = 0
                if any(term in process.name.lower() for term in ['loan', 'credit', 'account', 'bank', 'financial']):
                    banking_count += 1
                    score += 10
                if any(term in task_text for term in ['kyc', 'fraud', 'credit check', 'compliance', 'verification', 'disbursement']):
                    banking_count += 1
                    score += 10
                if any(term in resource_text for term in ['analyst', 'officer', 'kyc', 'fraud', 'compliance', 'loan', 'credit']):
                    banking_count += 1
                    score += 8
                if any(term in task_names_text for term in ['verification', 'approval', 'check', 'assessment', 'risk']):
                    banking_count += 1
                    score += 8
                
                if banking_count >= 3:
                    score += 20  # Strong boost for multiple banking indicators
                    reasons.append("Multiple strong banking indicators found")
                
                # Banking has approval gates
                if self._has_approval_gates(process):
                    score += 5
                    reasons.append("Approval workflow detected")
            
            scores[process_type] = score
            reasoning[process_type] = reasons
        
        # Determine the best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        total_score = sum(scores.values())
        
        # Calculate confidence
        confidence = best_score / max(total_score, 1) if total_score > 0 else 0.0
        
        # If confidence is too low, mark as unknown
        if confidence < 0.3 or best_score < 5:
            best_type = ProcessType.UNKNOWN
            confidence = 0.0
            best_reasons = ["No clear process type pattern detected"]
        else:
            best_reasons = reasoning[best_type]
        
        # Determine optimization strategy
        strategy = self._determine_strategy(best_type)
        
        # Get characteristics
        characteristics = self.patterns.get(
            best_type, 
            {'user_centric': False, 'sequential_flow': False, 
             'parallelism_level': 'medium', 'human_interaction': 'medium'}
        ).get('characteristics', {})
        
        return ProcessClassification(
            process_type=best_type,
            confidence=confidence,
            optimization_strategy=strategy,
            characteristics=characteristics,
            reasoning=best_reasons
        )
    
    def _analyze_text(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword match score"""
        if not text or not keywords:
            return 0.0
        
        matches = 0
        for keyword in keywords:
            if keyword in text:
                matches += 1
                # Count multiple occurrences
                matches += text.count(keyword) * 0.1
        
        return matches / len(keywords)
    
    def _analyze_dependency_structure(self, process: Process, process_type: ProcessType) -> float:
        """Analyze task dependency structure"""
        if not process.tasks:
            return 0.0
        
        total_tasks = len(process.tasks)
        tasks_with_deps = sum(1 for task in process.tasks if task.dependencies)
        
        if process_type == ProcessType.HEALTHCARE:
            # Healthcare tends to have sequential dependencies
            if tasks_with_deps / total_tasks > 0.6:
                return 5.0
        
        elif process_type == ProcessType.MANUFACTURING:
            # Manufacturing may have fewer dependencies (more parallel)
            if tasks_with_deps / total_tasks < 0.4:
                return 5.0
        
        elif process_type == ProcessType.BANKING:
            # Banking has moderate dependencies
            if 0.4 <= tasks_with_deps / total_tasks <= 0.7:
                return 3.0
        
        return 0.0
    
    def _has_sequential_flow(self, process: Process) -> bool:
        """Check if process has sequential flow pattern"""
        if not process.tasks:
            return False
        
        # Check if tasks form a chain (each depends on previous)
        tasks_by_id = {task.id: task for task in process.tasks}
        sequential_count = 0
        
        for task in process.tasks:
            if len(task.dependencies) == 1:
                dep_id = list(task.dependencies)[0]
                if dep_id in tasks_by_id:
                    sequential_count += 1
        
        return sequential_count / len(process.tasks) > 0.5
    
    def _has_high_parallelism(self, process: Process) -> bool:
        """Check if process has high parallelism potential"""
        if not process.tasks:
            return False
        
        # Tasks with no dependencies can run in parallel
        independent_tasks = sum(1 for task in process.tasks if not task.dependencies)
        return independent_tasks / len(process.tasks) > 0.3
    
    def _has_approval_gates(self, process: Process) -> bool:
        """Check if process has approval gate patterns"""
        approval_keywords = ['approval', 'approve', 'review', 'authorize', 'validate', 'verify']
        
        for task in process.tasks:
            task_text = f"{task.name} {task.description}".lower()
            if any(keyword in task_text for keyword in approval_keywords):
                # Check if this task is a dependency for many others (gate pattern)
                dependent_tasks = sum(
                    1 for other_task in process.tasks 
                    if task.id in other_task.dependencies
                )
                if dependent_tasks >= 2:
                    return True
        
        return False
    
    def _determine_strategy(self, process_type: ProcessType) -> OptimizationStrategy:
        """Determine the optimization strategy based on process type"""
        strategy_map = {
            ProcessType.HEALTHCARE: OptimizationStrategy.SEQUENTIAL_USER,
            ProcessType.MANUFACTURING: OptimizationStrategy.PARALLEL_PRODUCTION,
            ProcessType.BANKING: OptimizationStrategy.CONDITIONAL_APPROVAL,
            ProcessType.ACADEMIC: OptimizationStrategy.MIXED_ACADEMIC,
            ProcessType.UNKNOWN: OptimizationStrategy.BALANCED
        }
        
        return strategy_map.get(process_type, OptimizationStrategy.BALANCED)
    
    def get_optimization_parameters(self, classification: ProcessClassification) -> Dict[str, Any]:
        """
        Get optimization parameters based on process classification
        
        Returns:
            Dictionary of optimization parameters tailored to the process type
        """
        params = {
            'strategy': classification.optimization_strategy.value,
            'process_type': classification.process_type.value,
            'confidence': classification.confidence
        }
        
        if classification.optimization_strategy == OptimizationStrategy.SEQUENTIAL_USER:
            params.update({
                'minimize_user_waiting': True,
                'enforce_sequential': True,
                'parallelize_admin_tasks': True,
                'optimize_critical_path': True,
                'resource_continuity': True
            })
        
        elif classification.optimization_strategy == OptimizationStrategy.PARALLEL_PRODUCTION:
            params.update({
                'maximize_parallelism': True,
                'enforce_sequential': False,
                'optimize_throughput': True,
                'balance_workload': True,
                'minimize_idle_time': True
            })
        
        elif classification.optimization_strategy == OptimizationStrategy.CONDITIONAL_APPROVAL:
            params.update({
                'parallel_validations': True,
                'sequential_approvals': True,
                'optimize_approval_chain': True,
                'minimize_cycle_time': True,
                'track_compliance': True
            })
        
        elif classification.optimization_strategy == OptimizationStrategy.MIXED_ACADEMIC:
            params.update({
                'parallel_prerequisites': True,
                'coordinate_departments': True,
                'optimize_enrollment_flow': True,
                'balance_admin_load': True,
                'student_centric': True
            })
        
        else:  # BALANCED
            params.update({
                'minimize_user_waiting': True,
                'maximize_parallelism': False,
                'optimize_critical_path': True,
                'balance_workload': True,
                'enforce_sequential': False
            })
        
        return params
