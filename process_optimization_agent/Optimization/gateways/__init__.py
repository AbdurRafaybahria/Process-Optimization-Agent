"""
Gateway detection module for identifying parallel, exclusive, and inclusive gateway opportunities.
"""

from .gateway_base import GatewayDetectorBase, GatewayBranch, GatewaySuggestion
from .parallel_gateway_detector import ParallelGatewayDetector
from .exclusive_gateway_detector import ExclusiveGatewayDetector, DecisionPoint, ExclusiveBranch
from .inclusive_gateway_detector import InclusiveGatewayDetector

__all__ = [
    'GatewayDetectorBase',
    'GatewayBranch',
    'GatewaySuggestion',
    'ParallelGatewayDetector',
    'ExclusiveGatewayDetector',
    'InclusiveGatewayDetector',
    'DecisionPoint',
    'ExclusiveBranch'
]
