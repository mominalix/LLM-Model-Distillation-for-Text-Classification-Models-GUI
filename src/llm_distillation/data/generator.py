"""
Data generation module for creating synthetic datasets.

This module provides comprehensive data generation capabilities using LLMs
to create diverse, high-quality synthetic training data for text classification.
"""

import asyncio
import json
import re
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Iterator, Tuple
from pathlib import Path
import logging
import random

import pandas as pd
from tqdm import tqdm

from ..llm import BaseLLM, ModelManager, GenerationConfig
from ..config import Config, Language
from ..exceptions import DataGenerationError, ErrorCodes
from .validation import DataValidator, QualityMetrics
from .augmentation import DataAugmenter

logger = logging.getLogger(__name__)


@dataclass
class GenerationTask:
    """Configuration for a data generation task."""
    task_description: str
    class_labels: List[str]
    samples_per_class: int
    languages: List[Language] = field(default_factory=lambda: [Language.EN])
    temperature: float = 0.7
    max_retries: int = 3
    quality_threshold: float = 0.7
    enable_augmentation: bool = True
    enable_validation: bool = True
    output_format: str = "jsonl"
    
    def __post_init__(self):
        """Validate task configuration."""
        if not self.class_labels:
            raise ValueError("At least one class label is required")
        if self.samples_per_class < 1:
            raise ValueError("samples_per_class must be positive")
        if not self.languages:
            raise ValueError("At least one language is required")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if not 0.0 <= self.quality_threshold <= 1.0:
            raise ValueError("quality_threshold must be between 0.0 and 1.0")


@dataclass
class GenerationResult:
    """Result from data generation process."""
    task: GenerationTask
    generated_samples: List[Dict[str, Any]]
    quality_metrics: QualityMetrics
    generation_time: float
    total_cost: float
    total_tokens: int
    success_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_samples(self) -> int:
        return len(self.generated_samples)
    
    @property
    def samples_per_class_actual(self) -> Dict[str, int]:
        """Get actual sample counts per class."""
        counts = {}
        for sample in self.generated_samples:
            label = sample.get("label", "unknown")
            counts[label] = counts.get(label, 0) + 1
        return counts


class DataGenerator:
    """Main data generation orchestrator."""
    
    def __init__(
        self,
        config: Config,
        model_manager: ModelManager,
        validator: Optional[DataValidator] = None,
        augmenter: Optional[DataAugmenter] = None
    ):
        self.config = config
        self.model_manager = model_manager
        self.validator = validator or DataValidator(config)
        self.augmenter = augmenter or DataAugmenter(config)
        
        # Generation state
        self._current_task: Optional[GenerationTask] = None
        self._generation_progress: Dict[str, Any] = {}
        self._stop_generation = False
        
        # Callbacks for progress tracking
        self.progress_callback: Optional[Callable[[float, str], None]] = None
        self.sample_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        
        # Session tracking for saving partial datasets
        self._session_id: Optional[str] = None
    
    def generate_dataset(
        self,
        task: GenerationTask,
        model_name: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        sample_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> GenerationResult:
        """Generate a complete synthetic dataset."""
        
        self._current_task = task
        self.progress_callback = progress_callback
        self.sample_callback = sample_callback
        self._stop_generation = False
        
        # Create unique session ID for this generation
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        start_time = time.time()
        
        try:
            # Select model if not specified
            if not model_name:
                model_name = self.model_manager.select_model(
                    strategy="balanced",
                    task_type="data_generation"
                )
            
            model = self.model_manager.get_model(model_name)
            
            logger.info(f"Starting data generation for task: {task.task_description}")
            logger.info(f"Using model: {model_name}")
            logger.info(f"Target: {task.samples_per_class} samples per class")
            
            # Generate base samples
            self._update_progress(0.0, "Generating base samples...")
            base_samples = self._generate_base_samples(model, task)
            
            if self._stop_generation:
                # Save partial dataset and return partial result instead of error
                if base_samples:
                    saved_path = self._save_partial_dataset(base_samples, task, model_name, "stopped_after_generation")
                    logger.info(f"Saved partial dataset to: {saved_path}")
                    
                    # Return partial result instead of throwing error
                    generation_time = time.time() - start_time
                    return self._create_partial_result(
                        task, base_samples, model_name, generation_time, saved_path, "stopped_after_generation"
                    )
                else:
                    raise DataGenerationError("Generation stopped by user with no samples generated")
            
            # Validate samples
            self._update_progress(0.6, "Validating sample quality...")
            if task.enable_validation:
                validated_samples = self._validate_samples(base_samples, task)
            else:
                validated_samples = base_samples
            
            if self._stop_generation:
                # Save partial dataset and return partial result instead of error
                if validated_samples:
                    saved_path = self._save_partial_dataset(validated_samples, task, model_name, "stopped_after_validation")
                    logger.info(f"Saved partial dataset to: {saved_path}")
                    
                    # Return partial result instead of throwing error
                    generation_time = time.time() - start_time
                    return self._create_partial_result(
                        task, validated_samples, model_name, generation_time, saved_path, "stopped_after_validation"
                    )
                else:
                    raise DataGenerationError("Generation stopped by user with no samples generated")
            
            # Augment if needed
            self._update_progress(0.8, "Augmenting dataset...")
            final_samples = self._augment_if_needed(validated_samples, task)
            
            # Calculate metrics
            generation_time = time.time() - start_time
            quality_metrics = self._calculate_quality_metrics(final_samples)
            total_cost, total_tokens = self._calculate_costs(model)
            success_rate = self._calculate_success_rate(final_samples, task)
            
            self._update_progress(1.0, "Generation complete!")
            
            # Save completed dataset
            saved_path = self._save_partial_dataset(final_samples, task, model_name, "completed_successfully")
            
            result = GenerationResult(
                task=task,
                generated_samples=final_samples,
                quality_metrics=quality_metrics,
                generation_time=generation_time,
                total_cost=total_cost,
                total_tokens=total_tokens,
                success_rate=success_rate,
                metadata={
                    "model_used": model_name,
                    "generation_timestamp": time.time(),
                    "config_snapshot": self.config.to_dict(),
                    "saved_path": str(saved_path)
                }
            )
            
            logger.info(f"Generation completed: {len(final_samples)} samples in {generation_time:.2f}s")
            logger.info(f"Dataset saved to: {saved_path}")
            return result
            
        except Exception as e:
            logger.error(f"Data generation failed: {e}")
            raise DataGenerationError(
                message=f"Failed to generate dataset: {str(e)}",
                error_code=ErrorCodes.DATA_GENERATION_FAILED,
                details={
                    "task_description": task.task_description,
                    "model_name": model_name,
                    "elapsed_time": time.time() - start_time
                },
                original_error=e
            )
        finally:
            self._current_task = None
    
    def _generate_base_samples(
        self, 
        model: BaseLLM, 
        task: GenerationTask
    ) -> List[Dict[str, Any]]:
        """Generate base samples using the LLM for multiple languages."""
        
        all_samples = []
        # Total target now includes all languages
        total_target = len(task.class_labels) * task.samples_per_class * len(task.languages)
        
        # Generate in smaller batches to avoid token limits and truncation
        batch_size = min(25, task.samples_per_class)  # Reduced from 100 to 25
        
        # Generate for each language and class combination
        for lang_idx, language in enumerate(task.languages):
            if self._stop_generation:
                break
                
            logger.info(f"Generating samples for language: {language.value}")
            
            for class_idx, class_label in enumerate(task.class_labels):
                if self._stop_generation:
                    break
                    
                logger.info(f"Generating samples for class '{class_label}' in '{language.value}'")
                class_samples = []
                
                samples_needed = task.samples_per_class
                batch_count = 0
                
                # Calculate overall progress
                completed_samples = len(all_samples)
                progress = min(0.6 * completed_samples / total_target, 0.6)
                
                # Calculate current cost
                current_cost, _ = self._calculate_costs(model)
                
                self._update_progress(
                    progress,
                    f"Language '{language.value}' - Class '{class_label}' ({lang_idx + 1}/{len(task.languages)}, {class_idx + 1}/{len(task.class_labels)})",
                    current_samples=completed_samples,
                    total_target=total_target,
                    current_class=f"{language.value}: {class_label}",
                    class_progress=f"0/{samples_needed}",
                    current_cost=current_cost
                )
                
                while len(class_samples) < samples_needed and not self._stop_generation:
                    batch_count += 1
                    batch_samples_needed = min(batch_size, samples_needed - len(class_samples))
                    
                    # Update progress for batch start
                    current_cost, _ = self._calculate_costs(model)
                    completed_samples = len(all_samples) + len(class_samples)
                    
                    self._update_progress(
                        progress,
                        f"Generating batch {batch_count} for '{language.value}: {class_label}' ({len(class_samples)}/{samples_needed} samples)",
                        current_samples=completed_samples,
                        total_target=total_target,
                        current_class=f"{language.value}: {class_label}",
                        class_progress=f"{len(class_samples)}/{samples_needed}",
                        current_cost=current_cost
                    )
                    
                    # Create generation prompt with specific language
                    prompt = self._create_generation_prompt(
                        task, class_label, batch_samples_needed, language
                    )
                    
                    # Generate samples with progress tracking
                    # Show detailed progress before API call
                    self._update_progress(
                        progress,
                        f"API Request: '{language.value}: {class_label}' batch {batch_count} ({batch_samples_needed} samples)",
                        current_samples=completed_samples,
                        total_target=total_target,
                        current_class=f"{language.value}: {class_label}",
                        class_progress=f"{len(class_samples)}/{samples_needed} (requesting {batch_samples_needed})",
                        current_cost=current_cost
                    )
                    
                    batch_samples = self._generate_batch(
                        model, prompt, task, class_label, batch_samples_needed
                    )
                    
                    if batch_samples:
                        # Add language metadata to each sample
                        for sample in batch_samples:
                            sample['language'] = language.value
                        
                        class_samples.extend(batch_samples)
                        
                        # Update progress after successful batch
                        completed_samples = len(all_samples) + len(class_samples)
                        progress = min(0.6 * completed_samples / total_target, 0.6)
                        
                        # Calculate estimated cost so far
                        estimated_cost, _ = self._calculate_costs(model)
                        
                        self._update_progress(
                            progress,
                            f"'{language.value}: {class_label}': {len(class_samples)}/{samples_needed} samples | Cost: ${estimated_cost:.4f}",
                            current_samples=completed_samples,
                            total_target=total_target,
                            current_class=f"{language.value}: {class_label}",
                            class_progress=f"{len(class_samples)}/{samples_needed}",
                            current_cost=estimated_cost
                        )
                    
                    # Add delay to respect rate limits
                    time.sleep(0.1)
                
                all_samples.extend(class_samples)
                
                # Final update for completed class
                if len(class_samples) >= samples_needed:
                    completed_samples = len(all_samples)
                    final_cost, _ = self._calculate_costs(model)
                    
                    self._update_progress(
                        progress,
                        f"Completed '{language.value}: {class_label}': {len(class_samples)} samples generated!",
                        current_samples=completed_samples,
                        total_target=total_target,
                        current_class=f"{language.value}: {class_label}",
                        class_progress=f"{len(class_samples)}/{samples_needed} (COMPLETE)",
                        current_cost=final_cost
                    )
        
        logger.info(f"Generated {len(all_samples)} base samples")
        return all_samples
    
    def _save_partial_dataset(
        self,
        samples: List[Dict[str, Any]],
        task: GenerationTask,
        model_name: str,
        stop_reason: str
    ) -> Path:
        """Save partial dataset to a unique folder structure."""
        
        # Create unique folder name based on session and task
        # First, clean the task description by removing newlines and extra whitespace
        clean_description = re.sub(r'\s+', ' ', task.task_description.strip())
        task_name = clean_description[:50].replace(" ", "_")
        
        # Remove all invalid filename characters for Windows/Linux compatibility
        task_name = re.sub(r'[<>:"|?*\\/\n\r\t]', '_', task_name)
        
        # Remove any trailing dots or spaces that Windows doesn't like
        task_name = task_name.rstrip('._')
        
        # Ensure it's not empty
        if not task_name:
            task_name = "dataset"
        
        folder_name = f"{self._session_id}_{task_name}"
        dataset_path = self.config.datasets_dir / folder_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Save samples as JSONL
        samples_file = dataset_path / "samples.jsonl"
        with open(samples_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # Save metadata
        metadata = {
            "session_id": self._session_id,
            "created_at": datetime.now().isoformat(),
            "task_description": task.task_description,
            "class_labels": task.class_labels,
            "samples_per_class_target": task.samples_per_class,
            "languages": [lang.value for lang in task.languages],
            "temperature": task.temperature,
            "model_used": model_name,
            "stop_reason": stop_reason,
            "total_samples": len(samples),
            "samples_per_class_actual": self._count_samples_per_class(samples),
            "config_snapshot": self.config.to_dict()
        }
        
        metadata_file = dataset_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save task description as readable text
        task_file = dataset_path / "task_description.txt"
        with open(task_file, 'w', encoding='utf-8') as f:
            f.write(f"Task Description:\n{task.task_description}\n\n")
            f.write(f"Classes: {', '.join(task.class_labels)}\n")
            f.write(f"Target samples per class: {task.samples_per_class}\n")
            f.write(f"Languages: {', '.join([lang.value for lang in task.languages])}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Stop reason: {stop_reason}\n")
            f.write(f"Total samples generated: {len(samples)}\n")
        
        # Save class distribution summary
        class_counts = self._count_samples_per_class(samples)
        summary_file = dataset_path / "class_distribution.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Class Distribution:\n")
            f.write("==================\n")
            for class_label, count in class_counts.items():
                f.write(f"{class_label}: {count} samples\n")
            f.write(f"\nTotal: {sum(class_counts.values())} samples\n")
        
        logger.info(f"Saved partial dataset with {len(samples)} samples to: {dataset_path}")
        return dataset_path
    
    def _count_samples_per_class(self, samples: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count samples per class."""
        counts = {}
        for sample in samples:
            label = sample.get('label', 'unknown')
            counts[label] = counts.get(label, 0) + 1
        return counts
    
    def _create_partial_result(
        self,
        task: GenerationTask,
        samples: List[Dict[str, Any]],
        model_name: str,
        generation_time: float,
        saved_path: Path,
        stop_reason: str
    ) -> GenerationResult:
        """Create a GenerationResult for partial/stopped generation."""
        
        # Calculate basic metrics for partial result
        quality_metrics = self._calculate_quality_metrics(samples)
        total_cost, total_tokens = self._calculate_costs(self.model_manager.get_model(model_name))
        success_rate = self._calculate_success_rate(samples, task)
        
        # Update progress to reflect partial completion
        completion_ratio = len(samples) / (len(task.class_labels) * task.samples_per_class)
        self._update_progress(
            completion_ratio, 
            f"Generation stopped: {len(samples)} samples saved to {saved_path.name}"
        )
        
        result = GenerationResult(
            task=task,
            generated_samples=samples,
            quality_metrics=quality_metrics,
            generation_time=generation_time,
            total_cost=total_cost,
            total_tokens=total_tokens,
            success_rate=success_rate,
            metadata={
                "model_used": model_name,
                "generation_timestamp": time.time(),
                "config_snapshot": self.config.to_dict(),
                "saved_path": str(saved_path),
                "stop_reason": stop_reason,
                "is_partial_result": True,
                "completion_ratio": completion_ratio
            }
        )
        
        logger.info(f"Created partial result: {len(samples)} samples, stopped at {completion_ratio*100:.1f}% completion")
        return result
    
    def _create_generation_prompt(
        self,
        task: GenerationTask,
        class_label: str,
        batch_size: int,
        language: Language = None
    ) -> str:
        """Create a prompt for generating samples for a specific class and language."""
        
        language_descriptions = {
            Language.EN: "English",
            Language.AR: "Arabic",
            Language.ES: "Spanish", 
            Language.FR: "French",
            Language.ZH: "Chinese",
            Language.HI: "Hindi"
        }
        
        # Use specified language or default to first language in task
        target_language = language or task.languages[0]
        language_name = language_descriptions.get(target_language, target_language)
        
        prompt = f"""Generate {batch_size} diverse, high-quality text samples for a text classification task.

Task Description: {task.task_description}
Target Class: {class_label}
Language: {language_name}
All Available Classes: {', '.join(task.class_labels)}

Requirements:
1. Generate exactly {batch_size} unique text samples that clearly belong to the "{class_label}" class
2. Make each sample realistic and natural-sounding
3. Vary the length, style, and vocabulary across samples
4. Ensure samples are unambiguous and clearly distinguishable from other classes
5. Avoid bias, offensive content, or personally identifiable information
6. Include some challenging but still clearly classifiable examples

Format your response as a JSON array of objects, where each object has:
- "text": the sample text content
- "label": "{class_label}"
- "confidence": a score from 0.8 to 1.0 indicating how clearly this belongs to the class

Example format:
[
  {{"text": "example text here", "label": "{class_label}", "confidence": 0.95}},
  {{"text": "another example", "label": "{class_label}", "confidence": 0.9}}
]

Generate exactly {batch_size} samples:"""
        
        return prompt
    
    def _generate_batch(
        self,
        model: BaseLLM,
        prompt: str,
        task: GenerationTask,
        class_label: str,
        expected_count: int
    ) -> List[Dict[str, Any]]:
        """Generate a batch of samples with retry logic."""
        
        config = GenerationConfig(
            temperature=task.temperature,
            max_tokens=8000,  # Increased for larger JSON responses
            top_p=0.95
        )
        
        for attempt in range(task.max_retries):
            try:
                # Generate response
                response = model.generate_text(prompt, config)
                
                # Parse JSON response
                samples = self._parse_generation_response(
                    response.text, class_label, expected_count
                )
                
                if samples:
                    # Notify callback for each sample
                    if self.sample_callback:
                        for sample in samples:
                            self.sample_callback(sample)
                    
                    logger.debug(f"Generated {len(samples)} samples for {class_label}")
                    return samples
                
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == task.max_retries - 1:
                    # Last attempt failed, create fallback samples
                    return self._create_fallback_samples(class_label, expected_count)
        
        return []
    
    def _parse_generation_response(
        self,
        response_text: str,
        expected_label: str,
        expected_count: int
    ) -> List[Dict[str, Any]]:
        """Parse LLM response into structured samples."""
        
        try:
            # Try to find JSON in the response
            response_text = response_text.strip()
            
            # Find JSON array in response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']')
            
            logger.debug(f"JSON parsing: response_length={len(response_text)}, start_idx={start_idx}, end_idx={end_idx}")
            
            if start_idx == -1:
                logger.warning("No JSON array start bracket found in response")
                return []
            
            # Handle truncated JSON - try to extract valid objects even without closing bracket
            if end_idx == -1 or end_idx < start_idx:
                logger.warning("JSON array appears truncated, attempting to recover valid objects")
                # Find the last complete JSON object
                json_text = response_text[start_idx:]
                # Try to find the last complete object by looking for the last "},"
                last_complete = json_text.rfind('},')
                if last_complete != -1:
                    json_text = json_text[:last_complete + 1] + ']'  # Close the array
                    logger.debug(f"Recovered truncated JSON, length: {len(json_text)}")
                else:
                    logger.warning("Could not recover any complete objects from truncated response")
                    return []
            else:
                json_text = response_text[start_idx:end_idx + 1]
            
            samples_data = json.loads(json_text)
            
            if not isinstance(samples_data, list):
                logger.warning("Response is not a JSON array")
                return []
            
            # Process and validate samples
            valid_samples = []
            for item in samples_data:
                if not isinstance(item, dict):
                    continue
                
                text = item.get('text', '').strip()
                label = item.get('label', expected_label)
                confidence = item.get('confidence', 0.8)
                
                if text and len(text) >= self.config.min_text_length:
                    valid_samples.append({
                        'text': text,
                        'label': label,
                        'confidence': float(confidence),
                        'source': 'generated'
                    })
            
            logger.debug(f"Parsed {len(valid_samples)} valid samples from response")
            return valid_samples[:expected_count]  # Limit to expected count
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error parsing response: {e}")
            return []
    
    def _create_fallback_samples(
        self,
        class_label: str,
        count: int
    ) -> List[Dict[str, Any]]:
        """Create fallback samples when generation fails."""
        
        fallback_samples = []
        for i in range(count):
            fallback_samples.append({
                'text': f"Fallback sample {i+1} for {class_label} class",
                'label': class_label,
                'confidence': 0.5,
                'source': 'fallback'
            })
        
        logger.warning(f"Created {len(fallback_samples)} fallback samples for {class_label}")
        return fallback_samples
    
    def _validate_samples(
        self,
        samples: List[Dict[str, Any]],
        task: GenerationTask
    ) -> List[Dict[str, Any]]:
        """Validate and filter samples based on quality criteria."""
        
        if not task.enable_validation:
            return samples
        
        validated_samples = []
        
        for sample in samples:
            try:
                # Basic validation
                if not sample.get('text') or not sample.get('label'):
                    continue
                
                # Length validation
                text_length = len(sample['text'])
                if text_length < self.config.min_text_length or text_length > self.config.max_text_length:
                    continue
                
                # Quality validation using validator
                quality_score = self.validator.assess_sample_quality(sample)
                
                if quality_score >= task.quality_threshold:
                    sample['quality_score'] = quality_score
                    validated_samples.append(sample)
                
            except Exception as e:
                logger.warning(f"Error validating sample: {e}")
                continue
        
        logger.info(f"Validated {len(validated_samples)}/{len(samples)} samples")
        return validated_samples
    
    def _augment_if_needed(
        self,
        samples: List[Dict[str, Any]],
        task: GenerationTask
    ) -> List[Dict[str, Any]]:
        """Augment dataset if needed to reach target sample counts."""
        
        if not task.enable_augmentation:
            return samples
        
        # Check if we need more samples for any class
        current_counts = {}
        for sample in samples:
            label = sample['label']
            current_counts[label] = current_counts.get(label, 0) + 1
        
        samples_to_augment = []
        for label in task.class_labels:
            current = current_counts.get(label, 0)
            if current < task.samples_per_class:
                # Get samples for this class that can be augmented
                class_samples = [s for s in samples if s['label'] == label]
                if class_samples:
                    samples_to_augment.extend(
                        random.choices(
                            class_samples,
                            k=min(task.samples_per_class - current, len(class_samples) * 2)
                        )
                    )
        
        if samples_to_augment:
            logger.info(f"Augmenting {len(samples_to_augment)} samples")
            augmented = self.augmenter.augment_samples(samples_to_augment)
            samples.extend(augmented)
        
        return samples
    
    def _calculate_quality_metrics(self, samples: List[Dict[str, Any]]) -> QualityMetrics:
        """Calculate quality metrics for generated samples."""
        return self.validator.calculate_dataset_metrics(samples)
    
    def _calculate_costs(self, model: BaseLLM) -> Tuple[float, int]:
        """Calculate total costs and token usage."""
        stats = model.get_usage_stats()
        return stats.total_cost, stats.total_input_tokens + stats.total_output_tokens
    
    def _calculate_success_rate(
        self, 
        samples: List[Dict[str, Any]], 
        task: GenerationTask
    ) -> float:
        """Calculate generation success rate."""
        target_total = len(task.class_labels) * task.samples_per_class
        actual_total = len(samples)
        return min(actual_total / target_total, 1.0) if target_total > 0 else 0.0
    
    def _update_progress(self, progress: float, message: str, **kwargs) -> None:
        """Update progress and notify callback with detailed stats."""
        self._generation_progress = {
            'progress': progress,
            'message': message,
            'timestamp': time.time(),
            **kwargs
        }
        
        if self.progress_callback:
            self.progress_callback(progress, message)
        
        # Also update live stats if we have a main window reference
        if hasattr(self, '_main_window_ref') and self._main_window_ref:
            try:
                stats = {
                    'overall_progress': progress,
                    'current_samples': kwargs.get('current_samples', 0),
                    'total_target': kwargs.get('total_target', 0),
                    'current_class': kwargs.get('current_class', ''),
                    'class_progress': kwargs.get('class_progress', ''),
                    'current_cost': kwargs.get('current_cost', 0.0)
                }
                self._main_window_ref.root.after(0, 
                    lambda: self._main_window_ref.progress_panel.update_live_generation_stats(stats)
                )
            except Exception as e:
                # Ignore errors in UI updates
                pass
        
        logger.debug(f"Progress: {progress:.1%} - {message}")
    
    def stop_generation(self) -> None:
        """Stop the current generation process."""
        self._stop_generation = True
        logger.info("Generation stop requested")
    
    def get_generation_progress(self) -> Dict[str, Any]:
        """Get current generation progress."""
        return self._generation_progress.copy()
    
    def save_dataset(
        self,
        result: GenerationResult,
        output_path: Path,
        include_metadata: bool = True
    ) -> None:
        """Save generated dataset to file."""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if result.task.output_format == "jsonl":
            self._save_jsonl(result, output_path, include_metadata)
        elif result.task.output_format == "csv":
            self._save_csv(result, output_path, include_metadata)
        elif result.task.output_format == "parquet":
            self._save_parquet(result, output_path, include_metadata)
        else:
            raise ValueError(f"Unsupported output format: {result.task.output_format}")
        
        logger.info(f"Dataset saved to {output_path}")
    
    def _save_jsonl(
        self,
        result: GenerationResult,
        output_path: Path,
        include_metadata: bool
    ) -> None:
        """Save dataset in JSONL format."""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in result.generated_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        if include_metadata:
            metadata_path = output_path.with_suffix('.metadata.json')
            metadata = {
                'task': result.task.__dict__,
                'quality_metrics': result.quality_metrics.__dict__,
                'generation_stats': {
                    'total_samples': result.total_samples,
                    'generation_time': result.generation_time,
                    'total_cost': result.total_cost,
                    'success_rate': result.success_rate
                },
                'metadata': result.metadata
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
    
    def _save_csv(
        self,
        result: GenerationResult,
        output_path: Path,
        include_metadata: bool
    ) -> None:
        """Save dataset in CSV format."""
        df = pd.DataFrame(result.generated_samples)
        df.to_csv(output_path, index=False, encoding='utf-8')
    
    def _save_parquet(
        self,
        result: GenerationResult,
        output_path: Path,
        include_metadata: bool
    ) -> None:
        """Save dataset in Parquet format."""
        df = pd.DataFrame(result.generated_samples)
        df.to_parquet(output_path, index=False)
    
    async def generate_dataset_async(
        self,
        task: GenerationTask,
        model_name: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        sample_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> GenerationResult:
        """Asynchronously generate dataset (for GUI integration)."""
        
        # Run the synchronous generation in a thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self.generate_dataset,
                task,
                model_name,
                progress_callback,
                sample_callback
            )
            return await loop.run_in_executor(None, lambda: future.result())