"""
OpenAI API client implementation.

This module provides a concrete implementation of the BaseLLM interface
for OpenAI's API, including support for the latest 2025 models.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, AsyncIterator
import logging

import openai
from openai import AsyncOpenAI
import tiktoken

from .base import (
    BaseLLM, 
    ModelInfo, 
    GenerationConfig, 
    GenerationResult, 
    LLMProvider
)
from ..exceptions import APIError, ErrorCodes, handle_openai_error
from ..config import OpenAIModel

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLM):
    """OpenAI API client implementation."""
    
    # Model specifications updated for 2025
    MODEL_SPECS = {
        OpenAIModel.GPT_4_1_NANO: ModelInfo(
            name=OpenAIModel.GPT_4_1_NANO,
            provider=LLMProvider.OPENAI,
            context_length=128000,
            input_cost_per_1k=0.0001,
            output_cost_per_1k=0.0002,
            supports_streaming=True,
            supports_function_calling=True,
            max_tokens_per_minute=200000,
            description="Ultra-efficient nano model for high-volume tasks"
        ),
        OpenAIModel.GPT_4_1_MINI: ModelInfo(
            name=OpenAIModel.GPT_4_1_MINI,
            provider=LLMProvider.OPENAI,
            context_length=128000,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
            supports_streaming=True,
            supports_function_calling=True,
            max_tokens_per_minute=150000,
            description="Compact model with strong performance"
        ),
        OpenAIModel.GPT_4_1: ModelInfo(
            name=OpenAIModel.GPT_4_1,
            provider=LLMProvider.OPENAI,
            context_length=128000,
            input_cost_per_1k=0.03,
            output_cost_per_1k=0.06,
            supports_streaming=True,
            supports_function_calling=True,
            supports_vision=True,
            max_tokens_per_minute=40000,
            description="Full-capability model with advanced reasoning"
        ),
        OpenAIModel.GPT_4O: ModelInfo(
            name=OpenAIModel.GPT_4O,
            provider=LLMProvider.OPENAI,
            context_length=128000,
            input_cost_per_1k=0.0025,
            output_cost_per_1k=0.01,
            supports_streaming=True,
            supports_function_calling=True,
            supports_vision=True,
            max_tokens_per_minute=60000,
            description="Optimized for real-time applications"
        ),
        OpenAIModel.GPT_4O_MINI: ModelInfo(
            name=OpenAIModel.GPT_4O_MINI,
            provider=LLMProvider.OPENAI,
            context_length=128000,
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.0006,
            supports_streaming=True,
            supports_function_calling=True,
            max_tokens_per_minute=100000,
            description="Compact optimized model"
        ),
        OpenAIModel.O1_MINI: ModelInfo(
            name=OpenAIModel.O1_MINI,
            provider=LLMProvider.OPENAI,
            context_length=128000,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.012,
            supports_streaming=False,
            supports_function_calling=False,
            max_tokens_per_minute=20000,
            description="Reasoning model with chain-of-thought capabilities"
        ),
        OpenAIModel.O3_MINI: ModelInfo(
            name=OpenAIModel.O3_MINI,
            provider=LLMProvider.OPENAI,
            context_length=200000,
            input_cost_per_1k=0.005,
            output_cost_per_1k=0.015,
            supports_streaming=True,
            supports_function_calling=True,
            max_tokens_per_minute=30000,
            description="Latest generation compact model"
        ),
        OpenAIModel.O3_PRO: ModelInfo(
            name=OpenAIModel.O3_PRO,
            provider=LLMProvider.OPENAI,
            context_length=200000,
            input_cost_per_1k=0.02,
            output_cost_per_1k=0.08,
            supports_streaming=True,
            supports_function_calling=True,
            supports_vision=True,
            max_tokens_per_minute=10000,
            description="Professional-grade model with maximum capabilities"
        ),
    }
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        organization_id: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any
    ):
        super().__init__(model_name, api_key, **kwargs)
        
        self.organization_id = organization_id
        self.base_url = base_url
        
        # Initialize clients (following OpenAI documentation pattern)
        # OpenAI client will automatically read OPENAI_API_KEY from environment if api_key is not provided
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if organization_id:
            client_kwargs["organization"] = organization_id
        if base_url:
            client_kwargs["base_url"] = base_url
            
        self.client = openai.OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)
        
        # Initialize tokenizer
        try:
            # Use appropriate tokenizer for model
            if "gpt-4" in model_name.lower():
                self.tokenizer = tiktoken.encoding_for_model("gpt-4")
            else:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}, using fallback")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Validate connection on initialization
        if not self.validate_connection():
            raise APIError(
                message="Failed to validate OpenAI API connection",
                error_code=ErrorCodes.API_CONNECTION_FAILED
            )
    
    def generate_text(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any
    ) -> GenerationResult:
        """Generate text using OpenAI API."""
        if config is None:
            config = GenerationConfig()
        
        start_time = time.time()
        
        try:
            self._enforce_rate_limit()
            
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Prepare API parameters
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "frequency_penalty": config.frequency_penalty,
                "presence_penalty": config.presence_penalty,
                "stream": config.stream,
            }
            
            # Add optional parameters
            if config.max_tokens is not None:
                api_params["max_tokens"] = config.max_tokens
            if config.stop is not None:
                api_params["stop"] = config.stop
            if config.seed is not None:
                api_params["seed"] = config.seed
            if config.response_format is not None:
                api_params["response_format"] = config.response_format
            
            # Make API call
            response = self.client.chat.completions.create(**api_params)
            
            # Extract response data
            choice = response.choices[0]
            generated_text = choice.message.content or ""
            
            # Calculate metrics
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            latency = time.time() - start_time
            cost = self.estimate_cost(input_tokens, output_tokens)
            
            result = GenerationResult(
                text=generated_text,
                model=self.model_name,
                provider=LLMProvider.OPENAI,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost=cost,
                latency=latency,
                finish_reason=choice.finish_reason or "completed",
                metadata={
                    "response_id": response.id,
                    "created": response.created,
                    "system_fingerprint": getattr(response, "system_fingerprint", None),
                }
            )
            
            self._update_usage_stats(result)
            
            logger.info(
                f"Generated {output_tokens} tokens in {latency:.2f}s "
                f"(${cost:.4f})"
            )
            
            return result
            
        except Exception as e:
            self._handle_error(e)
            raise handle_openai_error(e)
    
    async def generate_text_async(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any
    ) -> GenerationResult:
        """Asynchronously generate text using OpenAI API."""
        if config is None:
            config = GenerationConfig()
        
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Prepare API parameters
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "frequency_penalty": config.frequency_penalty,
                "presence_penalty": config.presence_penalty,
                "stream": False,  # Not streaming for this method
            }
            
            # Add optional parameters
            if config.max_tokens is not None:
                api_params["max_tokens"] = config.max_tokens
            if config.stop is not None:
                api_params["stop"] = config.stop
            if config.seed is not None:
                api_params["seed"] = config.seed
            if config.response_format is not None:
                api_params["response_format"] = config.response_format
            
            # Make async API call
            response = await self.async_client.chat.completions.create(**api_params)
            
            # Extract response data
            choice = response.choices[0]
            generated_text = choice.message.content or ""
            
            # Calculate metrics
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            latency = time.time() - start_time
            cost = self.estimate_cost(input_tokens, output_tokens)
            
            result = GenerationResult(
                text=generated_text,
                model=self.model_name,
                provider=LLMProvider.OPENAI,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost=cost,
                latency=latency,
                finish_reason=choice.finish_reason or "completed",
                metadata={
                    "response_id": response.id,
                    "created": response.created,
                    "system_fingerprint": getattr(response, "system_fingerprint", None),
                }
            )
            
            self._update_usage_stats(result)
            return result
            
        except Exception as e:
            self._handle_error(e)
            raise handle_openai_error(e)
    
    async def generate_text_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream text generation using OpenAI API."""
        if config is None:
            config = GenerationConfig()
        
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Prepare API parameters
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "frequency_penalty": config.frequency_penalty,
                "presence_penalty": config.presence_penalty,
                "stream": True,
            }
            
            # Add optional parameters
            if config.max_tokens is not None:
                api_params["max_tokens"] = config.max_tokens
            if config.stop is not None:
                api_params["stop"] = config.stop
            if config.seed is not None:
                api_params["seed"] = config.seed
            
            # Make streaming API call
            stream = await self.async_client.chat.completions.create(**api_params)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self._handle_error(e)
            raise handle_openai_error(e)
    
    def get_model_info(self) -> ModelInfo:
        """Get information about the current model."""
        if self._model_info is None:
            self._model_info = self.MODEL_SPECS.get(
                self.model_name,
                ModelInfo(
                    name=self.model_name,
                    provider=LLMProvider.OPENAI,
                    context_length=128000,
                    input_cost_per_1k=0.001,
                    output_cost_per_1k=0.002,
                    description="Unknown OpenAI model"
                )
            )
        return self._model_info
    
    def validate_connection(self) -> bool:
        """Validate that the OpenAI API connection is working."""
        try:
            # Try to list models to validate API key
            models = self.client.models.list()
            
            # Check if our model is available
            available_models = [model.id for model in models.data]
            if self.model_name not in available_models:
                logger.warning(f"Model {self.model_name} not found in available models")
                # Don't fail validation - model might still work
            
            logger.info("OpenAI API connection validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"OpenAI API validation failed: {e}")
            return False
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the appropriate tokenizer."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using approximation")
            return super().count_tokens(text)
    
    def generate_system_prompt(
        self,
        task_description: str,
        class_labels: List[str],
        language: str = "en",
        examples_per_class: int = 1000
    ) -> str:
        """Generate a system prompt for data generation."""
        language_map = {
            "en": "English",
            "ar": "Arabic", 
            "es": "Spanish",
            "fr": "French",
            "zh": "Chinese",
            "hi": "Hindi"
        }
        
        language_name = language_map.get(language, language)
        
        prompt = f"""You are an expert data generator for machine learning tasks. Your job is to create high-quality, diverse synthetic text data for classification.

Task: {task_description}

Classes: {', '.join(class_labels)}
Language: {language_name}
Target: {examples_per_class} examples per class

Requirements:
1. Generate diverse, realistic text samples for each class
2. Ensure clear class distinction and avoid ambiguous examples
3. Include varied sentence structures, lengths, and vocabulary
4. Maintain consistent quality and relevance to the task
5. Avoid biased, offensive, or inappropriate content
6. Include edge cases and challenging examples

For each example, provide:
- The text content
- The correct class label
- Brief reasoning for the classification

Format your response as JSON with the following structure:
{{
    "examples": [
        {{
            "text": "example text here",
            "label": "class_name",
            "reasoning": "why this text belongs to this class"
        }}
    ]
}}

Generate exactly {examples_per_class} examples for each class: {', '.join(class_labels)}"""
        
        return prompt
    
    def batch_generate(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        max_concurrent: int = 5
    ) -> List[GenerationResult]:
        """Generate text for multiple prompts concurrently."""
        async def _batch_generate():
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def _generate_single(prompt: str) -> GenerationResult:
                async with semaphore:
                    return await self.generate_text_async(prompt, config)
            
            tasks = [_generate_single(prompt) for prompt in prompts]
            return await asyncio.gather(*tasks)
        
        # Run the async batch generation
        return asyncio.run(_batch_generate())
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available OpenAI models."""
        return list(cls.MODEL_SPECS.keys())
    
    @classmethod
    def get_model_recommendations(cls, use_case: str) -> List[str]:
        """Get recommended models for specific use cases."""
        recommendations = {
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
            "high_volume": [
                OpenAIModel.GPT_4_1_NANO,
                OpenAIModel.GPT_4O_MINI
            ],
            "quality": [
                OpenAIModel.O3_PRO,
                OpenAIModel.GPT_4_1,
                OpenAIModel.O3_MINI
            ]
        }
        
        return recommendations.get(use_case, [OpenAIModel.GPT_4_1_NANO])