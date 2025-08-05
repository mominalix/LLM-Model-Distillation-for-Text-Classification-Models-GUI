"""
LLM integration layer for the distillation application.

This module provides a unified interface for interacting with different
LLM providers, starting with OpenAI and designed for future extensibility.
"""

from .base import BaseLLM, ModelInfo, GenerationConfig
from .openai_client import OpenAIClient
from .model_manager import ModelManager

__all__ = [
    "BaseLLM",
    "ModelInfo", 
    "GenerationConfig",
    "OpenAIClient",
    "ModelManager",
]