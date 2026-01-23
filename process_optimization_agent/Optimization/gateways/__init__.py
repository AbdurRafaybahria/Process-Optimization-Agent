"""
Gateway detection module for identifying parallel, exclusive, and inclusive gateway opportunities.
"""

from .gateway_base import GatewayDetectorBase, GatewayBranch, GatewaySuggestion
from .parallel_gateway_detector import ParallelGatewayDetector
from .exclusive_gateway_detector import ExclusiveGatewayDetector, DecisionPoint, ExclusiveBranch

__all__ = [
    'GatewayDetectorBase',
    'GatewayBranch',
    'GatewaySuggestion',
    'ParallelGatewayDetector',
    'ExclusiveGatewayDetector',
    'DecisionPoint',
    'ExclusiveBranch'
]
