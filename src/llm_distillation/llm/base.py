"""
Abstract base classes for LLM integration.

This module defines the interface that all LLM providers must implement,
ensuring consistency and enabling easy swapping between different providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


@dataclass
class ModelInfo:
    """Information about an LLM model."""
    name: str
    provider: LLMProvider
    context_length: int
    input_cost_per_1k: float
    output_cost_per_1k: float
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    max_tokens_per_minute: Optional[int] = None
    description: Optional[str] = None


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    seed: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResult:
    """Result from text generation."""
    text: str
    model: str
    provider: LLMProvider
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    latency: float
    finish_reason: str
    metadata: Dict[str, Any]


@dataclass
class UsageStats:
    """Usage statistics for API calls."""
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    total_latency: float = 0.0
    error_count: int = 0
    last_call_time: Optional[float] = None


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs: Any
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.usage_stats = UsageStats()
        self._model_info: Optional[ModelInfo] = None
        
        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = kwargs.get("min_request_interval", 0.0)
        
        # Retry configuration
        self.max_retries = kwargs.get("max_retries", 3)
        self.retry_delay = kwargs.get("retry_delay", 1.0)
        
    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any
    ) -> GenerationResult:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    async def generate_text_async(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any
    ) -> GenerationResult:
        """Asynchronously generate text from a prompt."""
        pass
    
    @abstractmethod
    def generate_text_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream text generation."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Get information about the model."""
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate that the API connection is working."""
        pass
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Estimate cost for a request."""
        model_info = self.get_model_info()
        return (
            (input_tokens / 1000) * model_info.input_cost_per_1k +
            (output_tokens / 1000) * model_info.output_cost_per_1k
        )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text. Override in subclasses for accurate counting."""
        # Simple approximation: ~4 characters per token
        return len(text) // 4
    
    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        if self._min_request_interval > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < self._min_request_interval:
                sleep_time = self._min_request_interval - elapsed
                time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _update_usage_stats(self, result: GenerationResult) -> None:
        """Update usage statistics."""
        self.usage_stats.total_calls += 1
        self.usage_stats.total_input_tokens += result.input_tokens
        self.usage_stats.total_output_tokens += result.output_tokens
        self.usage_stats.total_cost += result.cost
        self.usage_stats.total_latency += result.latency
        self.usage_stats.last_call_time = time.time()
    
    def _handle_error(self, error: Exception) -> None:
        """Handle and log errors."""
        self.usage_stats.error_count += 1
        logger.error(f"LLM API error: {error}")
    
    def get_usage_stats(self) -> UsageStats:
        """Get usage statistics."""
        return self.usage_stats
    
    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.usage_stats = UsageStats()
    
    def get_average_latency(self) -> float:
        """Get average latency per request."""
        if self.usage_stats.total_calls == 0:
            return 0.0
        return self.usage_stats.total_latency / self.usage_stats.total_calls
    
    def get_average_cost_per_call(self) -> float:
        """Get average cost per API call."""
        if self.usage_stats.total_calls == 0:
            return 0.0
        return self.usage_stats.total_cost / self.usage_stats.total_calls
    
    def get_tokens_per_second(self) -> float:
        """Get average tokens generated per second."""
        if self.usage_stats.total_latency == 0:
            return 0.0
        return self.usage_stats.total_output_tokens / self.usage_stats.total_latency


class LLMPool:
    """Pool of LLM instances for load balancing and redundancy."""
    
    def __init__(self, llms: List[BaseLLM]):
        self.llms = llms
        self.current_index = 0
        
    def get_next_llm(self) -> BaseLLM:
        """Get next LLM instance using round-robin."""
        llm = self.llms[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.llms)
        return llm
    
    def get_least_used_llm(self) -> BaseLLM:
        """Get LLM instance with lowest usage."""
        return min(self.llms, key=lambda llm: llm.usage_stats.total_calls)
    
    def get_fastest_llm(self) -> BaseLLM:
        """Get LLM instance with lowest average latency."""
        valid_llms = [llm for llm in self.llms if llm.usage_stats.total_calls > 0]
        if not valid_llms:
            return self.llms[0]
        return min(valid_llms, key=lambda llm: llm.get_average_latency())
    
    def get_total_usage(self) -> UsageStats:
        """Get combined usage statistics."""
        total_stats = UsageStats()
        for llm in self.llms:
            stats = llm.usage_stats
            total_stats.total_calls += stats.total_calls
            total_stats.total_input_tokens += stats.total_input_tokens
            total_stats.total_output_tokens += stats.total_output_tokens
            total_stats.total_cost += stats.total_cost
            total_stats.total_latency += stats.total_latency
            total_stats.error_count += stats.error_count
        
        return total_stats