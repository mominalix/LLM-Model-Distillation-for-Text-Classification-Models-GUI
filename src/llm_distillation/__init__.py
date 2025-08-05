"""
LLM Model Distillation for Text Classification

A comprehensive GUI application that creates synthetic, class-balanced datasets
using OpenAI's latest models and distills their knowledge into small, efficient
Hugging Face models for multi-class text classification.
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"
__email__ = "team@airesearch.com"

from .config import Config, get_config
from .exceptions import (
    DistillationError,
    DataGenerationError,
    ModelLoadError,
    SecurityError,
)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "Config",
    "get_config", 
    "DistillationError",
    "DataGenerationError",
    "ModelLoadError",
    "SecurityError",
]