"""
Model manager for LLM selection and lifecycle management.

This module provides centralized management of LLM instances,
including model selection, caching, and optimization strategies.
"""

import time
from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
import logging

from .base import BaseLLM, ModelInfo, GenerationConfig, LLMProvider
from .openai_client import OpenAIClient
from ..config import Config, OpenAIModel, HuggingFaceModel
from ..exceptions import ModelLoadError, ErrorCodes

logger = logging.getLogger(__name__)


class ModelSelectionStrategy(str, Enum):
    """Strategies for selecting models."""
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    SPEED_OPTIMIZED = "speed_optimized"
    BALANCED = "balanced"


@dataclass
class ModelPerformance:
    """Performance metrics for a model."""
    average_latency: float = 0.0
    average_cost_per_call: float = 0.0
    tokens_per_second: float = 0.0
    error_rate: float = 0.0
    quality_score: float = 0.0
    last_updated: float = 0.0
    
    def update_metrics(self, llm: BaseLLM) -> None:
        """Update metrics from LLM usage stats."""
        stats = llm.get_usage_stats()
        self.average_latency = llm.get_average_latency()
        self.average_cost_per_call = llm.get_average_cost_per_call()
        self.tokens_per_second = llm.get_tokens_per_second()
        
        if stats.total_calls > 0:
            self.error_rate = stats.error_count / stats.total_calls
        
        self.last_updated = time.time()


class ModelManager:
    """Manages LLM instances and provides intelligent model selection."""
    
    def __init__(self, config: Config):
        self.config = config
        self._models: Dict[str, BaseLLM] = {}
        self._performance_metrics: Dict[str, ModelPerformance] = {}
        self._model_info_cache: Dict[str, ModelInfo] = {}
        
        # Initialize with default models
        self._initialize_default_models()
    
    def _initialize_default_models(self) -> None:
        """Initialize default model instances."""
        try:
            # Initialize default OpenAI model
            # Only pass organization_id if it's a valid value (not None, empty, or placeholder)
            org_id = self.config.openai_org_id
            if org_id and org_id.strip() and not org_id.startswith("your_"):
                org_id_to_use = org_id
            else:
                org_id_to_use = None
                
            default_openai = self.create_openai_client(
                self.config.default_teacher_model,
                self.config.openai_api_key,
                org_id_to_use
            )
            self.register_model(self.config.default_teacher_model, default_openai)
            
            logger.info(f"Initialized default teacher model: {self.config.default_teacher_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize default models: {e}")
    
    def create_openai_client(
        self,
        model_name: str,
        api_key: str,
        organization_id: Optional[str] = None,
        **kwargs: Any
    ) -> OpenAIClient:
        """Create an OpenAI client instance."""
        try:
            client = OpenAIClient(
                model_name=model_name,
                api_key=api_key,
                organization_id=organization_id,
                max_retries=self.config.max_workers,
                **kwargs
            )
            
            return client
            
        except Exception as e:
            raise ModelLoadError(
                message=f"Failed to create OpenAI client for {model_name}",
                error_code=ErrorCodes.MODEL_LOAD_FAILED,
                details={"model": model_name, "error": str(e)},
                original_error=e
            )
    
    def register_model(self, name: str, model: BaseLLM) -> None:
        """Register a model instance."""
        self._models[name] = model
        self._performance_metrics[name] = ModelPerformance()
        
        # Cache model info
        try:
            self._model_info_cache[name] = model.get_model_info()
        except Exception as e:
            logger.warning(f"Failed to cache model info for {name}: {e}")
        
        logger.info(f"Registered model: {name}")
    
    def unregister_model(self, name: str) -> None:
        """Unregister a model instance."""
        if name in self._models:
            del self._models[name]
        if name in self._performance_metrics:
            del self._performance_metrics[name]
        if name in self._model_info_cache:
            del self._model_info_cache[name]
        
        logger.info(f"Unregistered model: {name}")
    
    def get_model(self, name: str) -> BaseLLM:
        """Get a model instance by name."""
        if name not in self._models:
            # Try to create model on-demand
            if name in OpenAIModel.__members__.values():
                # Only pass organization_id if it's a valid value (not None, empty, or placeholder)
                org_id = self.config.openai_org_id
                if org_id and org_id.strip() and not org_id.startswith("your_"):
                    org_id_to_use = org_id
                else:
                    org_id_to_use = None
                    
                model = self.create_openai_client(
                    name,
                    self.config.openai_api_key,
                    org_id_to_use
                )
                self.register_model(name, model)
            else:
                raise ModelLoadError(
                    message=f"Model {name} not found and cannot be created",
                    error_code=ErrorCodes.MODEL_LOAD_FAILED,
                    details={"model": name}
                )
        
        return self._models[name]
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())
    
    def get_model_info(self, name: str) -> ModelInfo:
        """Get model information."""
        if name in self._model_info_cache:
            return self._model_info_cache[name]
        
        model = self.get_model(name)
        info = model.get_model_info()
        self._model_info_cache[name] = info
        return info
    
    def select_model(
        self,
        strategy: ModelSelectionStrategy = ModelSelectionStrategy.BALANCED,
        task_type: str = "generation",
        budget_limit: Optional[float] = None,
        latency_limit: Optional[float] = None,
        provider_preference: Optional[LLMProvider] = None
    ) -> str:
        """Select optimal model based on strategy and constraints."""
        
        # Filter available models
        candidates = self._filter_models(
            provider_preference=provider_preference,
            budget_limit=budget_limit,
            latency_limit=latency_limit
        )
        
        if not candidates:
            logger.warning("No models match criteria, using default")
            return self.config.default_teacher_model
        
        # Apply selection strategy
        if strategy == ModelSelectionStrategy.COST_OPTIMIZED:
            return self._select_cheapest_model(candidates)
        elif strategy == ModelSelectionStrategy.PERFORMANCE_OPTIMIZED:
            return self._select_highest_quality_model(candidates)
        elif strategy == ModelSelectionStrategy.SPEED_OPTIMIZED:
            return self._select_fastest_model(candidates)
        else:  # BALANCED
            return self._select_balanced_model(candidates, task_type)
    
    def _filter_models(
        self,
        provider_preference: Optional[LLMProvider] = None,
        budget_limit: Optional[float] = None,
        latency_limit: Optional[float] = None
    ) -> List[str]:
        """Filter models based on constraints."""
        candidates = []
        
        for name in self._models.keys():
            try:
                info = self.get_model_info(name)
                performance = self._performance_metrics[name]
                
                # Provider filter
                if provider_preference and info.provider != provider_preference:
                    continue
                
                # Budget filter
                if budget_limit and performance.average_cost_per_call > budget_limit:
                    continue
                
                # Latency filter
                if latency_limit and performance.average_latency > latency_limit:
                    continue
                
                candidates.append(name)
                
            except Exception as e:
                logger.warning(f"Error filtering model {name}: {e}")
                continue
        
        return candidates
    
    def _select_cheapest_model(self, candidates: List[str]) -> str:
        """Select model with lowest cost."""
        if not candidates:
            return self.config.default_teacher_model
        
        cheapest = candidates[0]
        cheapest_cost = float('inf')
        
        for name in candidates:
            try:
                info = self.get_model_info(name)
                # Use output cost as primary metric for generation tasks
                cost = info.output_cost_per_1k
                
                if cost < cheapest_cost:
                    cheapest_cost = cost
                    cheapest = name
                    
            except Exception as e:
                logger.warning(f"Error evaluating cost for {name}: {e}")
        
        return cheapest
    
    def _select_fastest_model(self, candidates: List[str]) -> str:
        """Select model with lowest latency."""
        if not candidates:
            return self.config.default_teacher_model
        
        fastest = candidates[0]
        fastest_latency = float('inf')
        
        for name in candidates:
            performance = self._performance_metrics[name]
            
            # Use tokens per second if available, otherwise latency
            if performance.tokens_per_second > 0:
                # Higher tokens/sec is better
                metric = 1.0 / performance.tokens_per_second
            else:
                metric = performance.average_latency
            
            if metric < fastest_latency:
                fastest_latency = metric
                fastest = name
        
        return fastest
    
    def _select_highest_quality_model(self, candidates: List[str]) -> str:
        """Select model with highest quality score."""
        if not candidates:
            return self.config.default_teacher_model
        
        # For now, use a simple heuristic based on model name
        # In production, this would use actual quality metrics
        quality_ranking = [
            OpenAIModel.O3_PRO,
            OpenAIModel.GPT_4_1,
            OpenAIModel.O3_MINI,
            OpenAIModel.GPT_4O,
            OpenAIModel.O1_MINI,
            OpenAIModel.GPT_4_1_MINI,
            OpenAIModel.GPT_4O_MINI,
            OpenAIModel.GPT_4_1_NANO,
        ]
        
        for model in quality_ranking:
            if model in candidates:
                return model
        
        return candidates[0]
    
    def _select_balanced_model(self, candidates: List[str], task_type: str) -> str:
        """Select model with best balance of cost, speed, and quality."""
        if not candidates:
            return self.config.default_teacher_model
        
        # Task-specific recommendations
        task_preferences = {
            "data_generation": [
                OpenAIModel.GPT_4_1_NANO,
                OpenAIModel.GPT_4_1_MINI,
                OpenAIModel.GPT_4O_MINI
            ],
            "reasoning": [
                OpenAIModel.O1_MINI,
                OpenAIModel.O3_MINI,
                OpenAIModel.GPT_4_1
            ],
            "evaluation": [
                OpenAIModel.GPT_4O,
                OpenAIModel.GPT_4_1,
                OpenAIModel.O3_MINI
            ]
        }
        
        preferred_models = task_preferences.get(task_type, [OpenAIModel.GPT_4_1_NANO])
        
        # Find first available preferred model
        for model in preferred_models:
            if model in candidates:
                return model
        
        # Fallback to cost optimization
        return self._select_cheapest_model(candidates)
    
    def update_performance_metrics(self, model_name: str) -> None:
        """Update performance metrics for a model."""
        if model_name in self._models and model_name in self._performance_metrics:
            model = self._models[model_name]
            metrics = self._performance_metrics[model_name]
            metrics.update_metrics(model)
    
    def get_performance_metrics(self, model_name: str) -> Optional[ModelPerformance]:
        """Get performance metrics for a model."""
        return self._performance_metrics.get(model_name)
    
    def get_cost_estimate(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Get cost estimate for a request."""
        try:
            model = self.get_model(model_name)
            return model.estimate_cost(input_tokens, output_tokens)
        except Exception as e:
            logger.warning(f"Error estimating cost for {model_name}: {e}")
            return 0.0
    
    def generate_with_fallback(
        self,
        prompt: str,
        preferred_model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        fallback_models: Optional[List[str]] = None
    ) -> Any:
        """Generate text with automatic fallback to alternative models."""
        models_to_try = []
        
        # Add preferred model first
        if preferred_model:
            models_to_try.append(preferred_model)
        
        # Add fallback models
        if fallback_models:
            models_to_try.extend(fallback_models)
        
        # Add default fallbacks
        models_to_try.extend([
            OpenAIModel.GPT_4_1_NANO,
            OpenAIModel.GPT_4O_MINI,
            OpenAIModel.GPT_4_1_MINI
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_models = []
        for model in models_to_try:
            if model not in seen:
                unique_models.append(model)
                seen.add(model)
        
        last_error = None
        
        for model_name in unique_models:
            try:
                model = self.get_model(model_name)
                result = model.generate_text(prompt, config)
                
                # Update metrics on successful generation
                self.update_performance_metrics(model_name)
                
                logger.info(f"Successfully generated text using {model_name}")
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to generate with {model_name}: {e}")
                continue
        
        # All models failed
        raise ModelLoadError(
            message="All fallback models failed",
            error_code=ErrorCodes.MODEL_LOAD_FAILED,
            details={"attempted_models": unique_models},
            original_error=last_error
        )
    
    def cleanup_unused_models(self, max_idle_time: float = 3600) -> None:
        """Clean up models that haven't been used recently."""
        current_time = time.time()
        models_to_remove = []
        
        for name, model in self._models.items():
            last_call_time = model.usage_stats.last_call_time
            
            if last_call_time and (current_time - last_call_time) > max_idle_time:
                models_to_remove.append(name)
        
        for name in models_to_remove:
            # Don't remove default model
            if name != self.config.default_teacher_model:
                self.unregister_model(name)
                logger.info(f"Cleaned up unused model: {name}")
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary for all models."""
        summary = {
            "total_models": len(self._models),
            "models": {}
        }
        
        for name, model in self._models.items():
            stats = model.get_usage_stats()
            performance = self._performance_metrics[name]
            
            summary["models"][name] = {
                "calls": stats.total_calls,
                "total_cost": stats.total_cost,
                "total_tokens": stats.total_input_tokens + stats.total_output_tokens,
                "error_count": stats.error_count,
                "average_latency": performance.average_latency,
                "tokens_per_second": performance.tokens_per_second,
            }
        
        return summary