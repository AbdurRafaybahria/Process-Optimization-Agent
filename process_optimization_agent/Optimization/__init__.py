"""
Process Optimization Agent - AI-powered project management system

This package provides intelligent task scheduling, resource management,
and process optimization using advanced AI techniques including
reinforcement learning and genetic algorithms.
"""

__version__ = "1.0.0"
__author__ = "Process Optimization Team"

try:
    from ..visualization import Visualizer, VISUALIZATION_AVAILABLE
except ImportError:
    class DummyVisualizer:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                print("[WARNING] Visualization is not available. Install required packages with: pip install matplotlib seaborn")
            return method
    Visualizer = DummyVisualizer()
    VISUALIZATION_AVAILABLE = False

# Import core optimization components
from .models import Task, Resource, Process, Schedule
from .optimizers import ProcessOptimizer, RLBasedOptimizer, GeneticOptimizer
from .analyzers import DependencyDetector, WhatIfAnalyzer, ProcessMiner
from .intelligent_optimizer import IntelligentOptimizer
from .cms_client import CMSClient
from .cms_transformer import CMSDataTransformer, ProcessValidationError
from .process_intelligence import ProcessIntelligence, ProcessType, OptimizationStrategy
from .user_journey_optimizer import UserJourneyOptimizer, UserJourneyMetrics
from .task_classifier import TaskClassifier
from .gateways import (
    ParallelGatewayDetector, 
    ExclusiveGatewayDetector,
    GatewayDetectorBase, 
    GatewayBranch, 
    GatewaySuggestion,
    DecisionPoint,
    ExclusiveBranch
)

# Import NLP dependency analyzer (optional)
try:
    from .nlp_dependency_analyzer import NLPDependencyAnalyzer, TaskRelationship, DependencyType, ConfidenceLevel
    NLP_ANALYZER_AVAILABLE = True
except ImportError:
    NLP_ANALYZER_AVAILABLE = False
    NLPDependencyAnalyzer = None
    TaskRelationship = None
    DependencyType = None
    ConfidenceLevel = None

# Domain-specific modules from scenarios
from ..scenarios.healthcare.healthcare_optimizer import HealthcareOptimizer
from ..scenarios.healthcare.healthcare_models import HealthcareProcess, HealthcareMetrics
from ..scenarios.manufacturing.manufacturing_optimizer import ManufacturingOptimizer
from ..scenarios.manufacturing.manufacturing_models import ManufacturingProcess, ManufacturingMetrics
from ..scenarios.insurance.insurance_optimizer import InsuranceProcessOptimizer
from ..scenarios.insurance.insurance_models import InsuranceProcess, InsuranceMetrics
# Banking module disabled
# from ..scenarios.banking.banking_optimizer import BankingProcessOptimizer
# from ..scenarios.banking.banking_models import BankingProcess, BankingMetrics
# from ..scenarios.banking.banking_detector import BankingProcessDetector

__all__ = [
    'Task', 'Resource', 'Process', 'Schedule',
    'ProcessOptimizer', 'RLBasedOptimizer', 'GeneticOptimizer',
    'DependencyDetector', 'WhatIfAnalyzer', 'ProcessMiner',
    'IntelligentOptimizer', 'CMSClient', 'CMSDataTransformer', 'ProcessValidationError',
    'ProcessIntelligence', 'ProcessType', 'OptimizationStrategy',
    'UserJourneyOptimizer', 'UserJourneyMetrics', 'TaskClassifier',
    'ParallelGatewayDetector', 'GatewaySuggestion',
    'GatewayDetectorBase', 'GatewayBranch',
    'ExclusiveGatewayDetector', 'DecisionPoint', 'ExclusiveBranch',
    'Visualizer', 'VISUALIZATION_AVAILABLE',
    # NLP components
    'NLPDependencyAnalyzer', 'TaskRelationship', 'DependencyType', 'ConfidenceLevel', 'NLP_ANALYZER_AVAILABLE',
    # Domain-specific
    'HealthcareOptimizer', 'HealthcareProcess', 'HealthcareMetrics',
    'ManufacturingOptimizer', 'ManufacturingProcess', 'ManufacturingMetrics',
    'InsuranceProcessOptimizer', 'InsuranceProcess', 'InsuranceMetrics'
    # Banking disabled: 'BankingProcessOptimizer', 'BankingProcess', 'BankingMetrics', 'BankingProcessDetector'
]
