"""
GUI module for the LLM Distillation application.

This module provides a comprehensive graphical user interface using CustomTkinter
for creating synthetic datasets and training distilled models.
"""

from .main_window import MainWindow
from .components import *
from .utils import *

__all__ = [
    "MainWindow",
]