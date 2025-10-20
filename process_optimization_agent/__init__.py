"""
Process Optimization Agent - AI-powered project management system

This package provides intelligent task scheduling, resource management,
and process optimization using advanced AI techniques including
reinforcement learning and genetic algorithms.
"""

__version__ = "1.0.0"
__author__ = "Process Optimization Team"

try:
    from .visualizer import Visualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    class DummyVisualizer:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                print("[WARNING] Visualization is not available. Install required packages with: pip install matplotlib seaborn")
            return method
    Visualizer = DummyVisualizer()
    VISUALIZATION_AVAILABLE = False

from .models import Task, Resource, Process, Schedule
from .optimizers import ProcessOptimizer, RLBasedOptimizer, GeneticOptimizer
from .analyzers import DependencyDetector, WhatIfAnalyzer, ProcessMiner

# Domain-specific modules
from .healthcare_optimizer import HealthcareOptimizer
from .healthcare_models import HealthcareProcess, HealthcareMetrics
from .manufacturing_optimizer import ManufacturingOptimizer
from .manufacturing_models import ManufacturingProcess, ManufacturingMetrics
from .banking_optimizer import BankingProcessOptimizer
from .banking_models import BankingProcess
from .banking_detector import BankingProcessDetector

__all__ = [
    'Task', 'Resource', 'Process', 'Schedule',
    'ProcessOptimizer', 'RLBasedOptimizer', 'GeneticOptimizer',
    'DependencyDetector', 'WhatIfAnalyzer', 'ProcessMiner',
    'Visualizer', 'VISUALIZATION_AVAILABLE',
    # Domain-specific
    'HealthcareOptimizer', 'HealthcareProcess', 'HealthcareMetrics',
    'ManufacturingOptimizer', 'ManufacturingProcess', 'ManufacturingMetrics',
    'BankingProcessOptimizer', 'BankingProcess', 'BankingProcessDetector'
]
