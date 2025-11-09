"""Visualization module for process optimization"""

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

__all__ = ['Visualizer', 'VISUALIZATION_AVAILABLE']
