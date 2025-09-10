"""
Active Learning Molecular Docking Package

Top journal-quality implementation of active learning for molecular docking
with TabM deep learning model.
"""

__version__ = "1.0.0"
__author__ = "Active Learning Molecular Docking Team" 
__license__ = "MIT"
__description__ = "Active learning framework for molecular docking with TabM model"

# Main classes
from .Activedocking import ActiveLearningWithConformers
from .unified_model import UnifiedModel

__all__ = [
    '__version__',
    '__author__',
    '__license__',
    '__description__',
    'ActiveLearningWithConformers',
    'UnifiedModel'
]