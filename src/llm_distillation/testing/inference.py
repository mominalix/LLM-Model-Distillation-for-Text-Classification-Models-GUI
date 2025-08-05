"""
Model inference and testing functionality.

This module provides comprehensive model testing capabilities including
single sample inference, batch processing, and performance evaluation.
"""

import os
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import Dataset

from ..config import Config
from ..training.evaluation import ModelEvaluator, EvaluationMetrics
from ..exceptions import DistillationError, ErrorCodes

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    
    # Model configuration
    model_path: str = ""
    use_gpu: bool = True
    batch_size: int = 32
    max_length: int = 512
    
    # Inference settings
    return_probabilities: bool = True
    return_confidence: bool = True
    top_k_predictions: int = 3
    
    # Performance settings
    use_torch_compile: bool = False
    use_mixed_precision: bool = True
    

@dataclass
class InferenceResult:
    """Result from model inference."""
    
    # Input information
    text: str = ""
    
    # Predictions
    predicted_label: str = ""
    predicted_class_id: int = -1
    confidence: float = 0.0
    
    # Detailed results
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    top_k_predictions: List[Tuple[str, float]] = field(default_factory=list)
    
    # Performance metrics
    inference_time_ms: float = 0.0
    
    # Model information
    model_name: str = ""
    model_path: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'text': self.text,
            'predicted_label': self.predicted_label,
            'predicted_class_id': self.predicted_class_id,
            'confidence': self.confidence,
            'class_probabilities': self.class_probabilities,
            'top_k_predictions': self.top_k_predictions,
            'inference_time_ms': self.inference_time_ms,
            'model_name': self.model_name,
            'model_path': self.model_path
        }


class ModelTester:
    """Comprehensive model testing and inference system."""
    
    def __init__(
        self,
        config: Config,
        inference_config: Optional[InferenceConfig] = None
    ):
        self.config = config
        self.inference_config = inference_config or InferenceConfig()
        
        # Model components
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.pipeline: Optional[pipeline] = None
        
        # Model metadata
        self.class_names: List[str] = []
        self.model_info: Dict[str, Any] = {}
        
        # Device configuration
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.inference_config.use_gpu else "cpu"
        )
        
        logger.info(f"ModelTester initialized on device: {self.device}")
    
    def load_model(self, model_path: Union[str, Path]) -> bool:
        """Load a trained model for inference."""
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
            logger.info(f"Loading model from: {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info("Tokenizer loaded successfully")
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.inference_config.use_mixed_precision else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Enable torch.compile if requested (PyTorch 2.0+)
            if self.inference_config.use_torch_compile:
                try:
                    self.model = torch.compile(self.model)
                    logger.info("Model compiled with torch.compile")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")
            
            logger.info("Model loaded successfully")
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                top_k=None  # Return all scores (replaces deprecated return_all_scores)
            )
            
            # Load model metadata
            self._load_model_metadata(model_path)
            
            # Extract class names from model config
            self._extract_class_names()
            
            logger.info(f"Model loaded with {len(self.class_names)} classes: {self.class_names}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            return False
    
    def _load_model_metadata(self, model_path: Path) -> None:
        """Load model metadata from various config files."""
        self.model_info = {
            'model_path': str(model_path),
            'model_name': model_path.name,
            'load_timestamp': time.time()
        }
        
        # Try to load export metadata
        metadata_file = model_path / "export_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    export_metadata = json.load(f)
                self.model_info.update(export_metadata)
            except Exception as e:
                logger.warning(f"Failed to load export metadata: {e}")
        
        # Try to load distillation config
        distillation_config_file = model_path / "distillation_config.json"
        if distillation_config_file.exists():
            try:
                with open(distillation_config_file, 'r') as f:
                    distillation_config = json.load(f)
                self.model_info['distillation_config'] = distillation_config
            except Exception as e:
                logger.warning(f"Failed to load distillation config: {e}")
    
    def _extract_class_names(self) -> None:
        """Extract class names from model configuration."""
        if self.model and hasattr(self.model.config, 'id2label'):
            self.class_names = [
                self.model.config.id2label[i] 
                for i in sorted(self.model.config.id2label.keys())
            ]
        else:
            # Fallback to numeric labels
            num_labels = getattr(self.model.config, 'num_labels', 2)
            self.class_names = [f"Class_{i}" for i in range(num_labels)]
    
    def predict_single(self, text: str) -> InferenceResult:
        """Perform inference on a single text sample."""
        if not self.model or not self.tokenizer:
            raise DistillationError(
                message="No model loaded. Please load a model first.",
                error_code=ErrorCodes.MODEL_NOT_FOUND
            )
        
        start_time = time.time()
        
        try:
            # Use pipeline for inference
            pipeline_results = self.pipeline(text)
            
            logger.info(f"Pipeline output type: {type(pipeline_results)}")
            logger.info(f"Pipeline output: {pipeline_results}")
            
            # Handle different result formats
            if isinstance(pipeline_results, list) and len(pipeline_results) > 0:
                # Check if it's a list of predictions for one sample
                if isinstance(pipeline_results[0], dict) and 'label' in pipeline_results[0]:
                    # This is the expected format: list of {'label': 'X', 'score': Y}
                    results = pipeline_results
                elif isinstance(pipeline_results[0], list):
                    # This is a list of lists (batch output), take first sample
                    results = pipeline_results[0]
                else:
                    results = pipeline_results
            else:
                # Single result or unexpected format
                results = [pipeline_results] if not isinstance(pipeline_results, list) else pipeline_results
            
            logger.info(f"Processed results: {results}")
            
            # Validate result format
            if not results or not isinstance(results[0], dict) or 'label' not in results[0]:
                raise ValueError(f"Unexpected pipeline output format: {results}")
            
            # Extract predictions
            class_probs = {result['label']: result['score'] for result in results}
            
            # Get top prediction
            top_result = max(results, key=lambda x: x['score'])
            predicted_label = top_result['label']
            confidence = top_result['score']
            
            # Get predicted class ID
            predicted_class_id = self.class_names.index(predicted_label) if predicted_label in self.class_names else -1
            
            # Get top-k predictions
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            top_k_predictions = [
                (result['label'], result['score']) 
                for result in sorted_results[:self.inference_config.top_k_predictions]
            ]
            
            inference_time_ms = (time.time() - start_time) * 1000
            
            return InferenceResult(
                text=text,
                predicted_label=predicted_label,
                predicted_class_id=predicted_class_id,
                confidence=confidence,
                class_probabilities=class_probs,
                top_k_predictions=top_k_predictions,
                inference_time_ms=inference_time_ms,
                model_name=self.model_info.get('model_name', 'Unknown'),
                model_path=self.model_info.get('model_path', '')
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise DistillationError(
                message=f"Failed to perform inference: {str(e)}",
                error_code=ErrorCodes.TRAINING_FAILED
            )
    
    def predict_batch(
        self, 
        texts: List[str],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[InferenceResult]:
        """Perform inference on multiple text samples."""
        if not self.model or not self.tokenizer:
            raise DistillationError(
                message="No model loaded. Please load a model first.",
                error_code=ErrorCodes.MODEL_NOT_FOUND
            )
        
        results = []
        total_samples = len(texts)
        
        logger.info(f"Starting batch inference on {total_samples} samples")
        
        # Process in batches
        batch_size = self.inference_config.batch_size
        
        for i in range(0, total_samples, batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Update progress
            if progress_callback:
                progress = i / total_samples
                progress_callback(progress, f"Processing batch {i//batch_size + 1}")
            
            # Process batch
            batch_results = []
            for text in batch_texts:
                try:
                    result = self.predict_single(text)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process text: {text[:50]}... Error: {e}")
                    # Create error result
                    error_result = InferenceResult(
                        text=text,
                        predicted_label="ERROR",
                        confidence=0.0,
                        model_name=self.model_info.get('model_name', 'Unknown'),
                        model_path=self.model_info.get('model_path', '')
                    )
                    batch_results.append(error_result)
            
            results.extend(batch_results)
        
        if progress_callback:
            progress_callback(1.0, f"Completed {total_samples} predictions")
        
        logger.info(f"Batch inference completed. Processed {len(results)} samples")
        
        return results
    
    def evaluate_on_dataset(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> EvaluationMetrics:
        """Evaluate model performance on a labeled dataset."""
        if not self.model or not self.tokenizer:
            raise DistillationError(
                message="No model loaded. Please load a model first.",
                error_code=ErrorCodes.MODEL_NOT_FOUND
            )
        
        # Convert to dataset format if needed
        if isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)
        
        # Use the existing evaluator
        evaluator = ModelEvaluator(self.config)
        
        logger.info(f"Evaluating model on {len(dataset)} samples")
        
        metrics = evaluator.evaluate_model(
            model=self.model,
            tokenizer=self.tokenizer,
            test_dataset=dataset,
            include_llm_jury=False  # Skip LLM jury for faster evaluation
        )
        
        logger.info(f"Evaluation completed: {metrics.summary()}")
        
        return metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = dict(self.model_info)
        
        if self.model:
            info.update({
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'num_classes': len(self.class_names),
                'class_names': self.class_names,
                'device': str(self.device),
                'model_config': self.model.config.to_dict() if hasattr(self.model.config, 'to_dict') else str(self.model.config)
            })
        
        return info
    
    def get_available_models(self, models_dir: Union[str, Path] = None) -> List[Dict[str, Any]]:
        """Get list of available trained models."""
        if models_dir is None:
            models_dir = self.config.models_dir
        
        models_dir = Path(models_dir)
        
        if not models_dir.exists():
            logger.warning(f"Models directory does not exist: {models_dir}")
            return []
        
        available_models = []
        
        # Check for export metadata first (indicates exported models with multiple formats)
        export_metadata_file = models_dir / "export_metadata.json"
        if export_metadata_file.exists():
            try:
                with open(export_metadata_file, 'r') as f:
                    export_metadata = json.load(f)
                
                # Add each format as a separate option
                for format_name in export_metadata.get('formats', []):
                    format_path = models_dir / format_name
                    if format_path.exists() and self._is_valid_model_directory(format_path):
                        model_info = self._get_model_directory_info(format_path)
                        model_info.update({
                            'task_name': export_metadata.get('task_name', 'Unknown'),
                            'model_type': export_metadata.get('model_name', 'Unknown'),
                            'export_timestamp': export_metadata.get('export_timestamp'),
                            'format': format_name
                        })
                        available_models.append(model_info)
            except Exception as e:
                logger.warning(f"Failed to load export metadata: {e}")
        
        # Look for individual model directories
        for model_path in models_dir.iterdir():
            if model_path.is_dir():
                # Skip if already processed via export metadata
                if any(model['path'] == str(model_path) for model in available_models):
                    continue
                    
                # Check if it contains model files
                if self._is_valid_model_directory(model_path):
                    model_info = self._get_model_directory_info(model_path)
                    available_models.append(model_info)
        
        # Also check output directory for recent training results
        output_dir = Path(self.config.output_dir)
        if output_dir.exists():
            training_dir = output_dir / "training"
            if training_dir.exists():
                for model_path in training_dir.rglob("pytorch_model.bin"):
                    parent_dir = model_path.parent
                    if self._is_valid_model_directory(parent_dir):
                        model_info = self._get_model_directory_info(parent_dir)
                        model_info['source'] = 'training_output'
                        available_models.append(model_info)
        
        # Sort by modification time (newest first)
        available_models.sort(key=lambda x: x.get('last_modified', 0), reverse=True)
        
        logger.info(f"Found {len(available_models)} available models")
        
        return available_models
    
    def _is_valid_model_directory(self, model_path: Path) -> bool:
        """Check if directory contains a valid model."""
        required_files = ['config.json']
        model_files = ['pytorch_model.bin', 'model.safetensors']
        tokenizer_files = ['tokenizer.json', 'tokenizer_config.json']
        
        # Check for required config
        if not any((model_path / f).exists() for f in required_files):
            return False
        
        # Check for model files
        if not any((model_path / f).exists() for f in model_files):
            return False
        
        # Optionally check for tokenizer (not required for all models)
        return True
    
    def _get_model_directory_info(self, model_path: Path) -> Dict[str, Any]:
        """Extract information about a model directory."""
        info = {
            'name': model_path.name,
            'path': str(model_path),
            'last_modified': model_path.stat().st_mtime,
            'size_mb': sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024 * 1024)
        }
        
        # Try to load export metadata
        metadata_file = model_path / "export_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                info.update({
                    'task_name': metadata.get('task_name', 'Unknown'),
                    'model_type': metadata.get('model_name', 'Unknown'),
                    'export_timestamp': metadata.get('export_timestamp'),
                    'formats': metadata.get('formats', [])
                })
            except Exception as e:
                logger.warning(f"Failed to load metadata for {model_path}: {e}")
        
        # Try to load model config for basic info
        config_file = model_path / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                info.update({
                    'num_labels': config.get('num_labels', 'Unknown'),
                    'model_architecture': config.get('architectures', ['Unknown'])[0] if config.get('architectures') else 'Unknown'
                })
            except Exception as e:
                logger.warning(f"Failed to load config for {model_path}: {e}")
        
        return info
    
    def unload_model(self) -> None:
        """Unload the current model to free memory."""
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.class_names = []
        self.model_info = {}
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded successfully")