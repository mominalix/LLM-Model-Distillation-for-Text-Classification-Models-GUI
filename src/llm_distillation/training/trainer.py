"""
General model training orchestration and management.

This module provides high-level training interfaces that coordinate
data preparation, model training, evaluation, and export.
"""

import os
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
import logging

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from ..config import Config
from ..llm import ModelManager
from ..data import DataProcessor, DatasetProcessor, DataValidator, GenerationResult
from .distillation import DistillationTrainer, DistillationConfig
from .evaluation import ModelEvaluator, EvaluationMetrics
from ..exceptions import TrainingError, ErrorCodes

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """High-level training configuration."""
    
    # Task configuration
    task_name: str = "text_classification"
    model_name: str = "distilbert-base-uncased"
    output_dir: str = "./training_output"
    
    # Data configuration
    train_split_ratio: float = 0.8
    val_split_ratio: float = 0.1
    test_split_ratio: float = 0.1
    max_sequence_length: int = 512
    
    # Training parameters
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Distillation parameters
    use_distillation: bool = True
    teacher_model: Optional[str] = None
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7
    
    # Evaluation configuration
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True
    
    # Hardware configuration
    use_mixed_precision: str = "bf16"
    dataloader_num_workers: int = 4
    
    # Export configuration
    export_formats: List[str] = field(default_factory=lambda: ["transformers", "safetensors"])
    export_onnx: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if not (0.0 < self.train_split_ratio < 1.0):
            raise ValueError("train_split_ratio must be between 0.0 and 1.0")
        
        total_ratio = self.train_split_ratio + self.val_split_ratio + self.test_split_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")


class ModelTrainer:
    """High-level model training orchestrator."""
    
    def __init__(
        self,
        config: Config,
        training_config: TrainingConfig,
        model_manager: Optional[ModelManager] = None
    ):
        self.config = config
        self.training_config = training_config
        self.model_manager = model_manager or ModelManager(config)
        
        # Initialize components
        self.data_processor = DataProcessor(config)
        self.dataset_processor = DatasetProcessor(config)
        self.validator = DataValidator(config)
        self.evaluator = ModelEvaluator(config)
        
        # Training state
        self.tokenizer: Optional[AutoTokenizer] = None
        self.dataset: Optional[DatasetDict] = None
        self.distillation_trainer: Optional[DistillationTrainer] = None
        
        # Callbacks
        self.progress_callback: Optional[Callable[[float, str], None]] = None
        self.metrics_callback: Optional[Callable[[Dict[str, float]], None]] = None
        
        # Training history
        self.training_history: List[Dict[str, Any]] = []
        
        logger.info(f"ModelTrainer initialized for task: {training_config.task_name}")
    
    def prepare_data(
        self,
        data_source: Union[List[Dict[str, Any]], GenerationResult, str, Path],
        validate_data: bool = True
    ) -> DatasetDict:
        """Prepare training data from various sources."""
        
        logger.info("Preparing training data")
        
        # Load data based on source type
        if isinstance(data_source, GenerationResult):
            # From synthetic data generation
            samples = data_source.generated_samples
            logger.info(f"Using synthetic data: {len(samples)} samples")
            
        elif isinstance(data_source, list):
            # Direct list of samples
            samples = data_source
            logger.info(f"Using provided samples: {len(samples)} samples")
            
        elif isinstance(data_source, (str, Path)):
            # From file
            samples = self.data_processor.load_dataset_from_file(data_source)
            logger.info(f"Loaded from file: {len(samples)} samples")
            
        else:
            raise TrainingError(
                message=f"Unsupported data source type: {type(data_source)}",
                error_code=ErrorCodes.DATA_GENERATION_FAILED
            )
        
        # Validate data quality
        if validate_data:
            logger.info("Validating data quality")
            
            # Filter low-quality samples
            samples = self.validator.filter_low_quality_samples(
                samples,
                quality_threshold=self.config.quality_threshold
            )
            
            # Calculate quality metrics
            quality_metrics = self.validator.calculate_dataset_metrics(samples)
            logger.info(f"Data quality metrics: {quality_metrics.overall_quality:.3f}")
        
        # Preprocess samples
        from ..data.processor import PreprocessingConfig
        preprocessing_config = PreprocessingConfig(
            lowercase=True,
            normalize_whitespace=True,
            remove_empty=True,
            min_length=self.config.min_text_length,
            max_length=self.config.max_text_length,
            remove_duplicates=True,
            balance_classes=False  # We'll handle this separately if needed
        )
        
        samples = self.data_processor.preprocess_samples(samples, preprocessing_config)
        
        # Split data
        from ..data.processor import DatasetSplit
        split_config = DatasetSplit(
            train_ratio=self.training_config.train_split_ratio,
            val_ratio=self.training_config.val_split_ratio,
            test_ratio=self.training_config.test_split_ratio,
            stratify=True,
            random_state=42
        )
        
        train_samples, val_samples, test_samples = self.data_processor.split_dataset(
            samples, split_config
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.training_config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare datasets for training
        train_dataset = self.dataset_processor.prepare_for_training(
            train_samples,
            tokenizer=self.tokenizer,
            max_length=self.training_config.max_sequence_length
        )['train']
        
        val_dataset = None
        if val_samples:
            val_dataset = self.dataset_processor.prepare_for_training(
                val_samples,
                tokenizer=self.tokenizer,
                max_length=self.training_config.max_sequence_length
            )['train']
        
        test_dataset = None
        if test_samples:
            test_dataset = self.dataset_processor.prepare_for_training(
                test_samples,
                tokenizer=self.tokenizer,
                max_length=self.training_config.max_sequence_length
            )['train']
        
        # Create dataset dictionary
        dataset_dict = {'train': train_dataset}
        if val_dataset:
            dataset_dict['validation'] = val_dataset
        if test_dataset:
            dataset_dict['test'] = test_dataset
        
        self.dataset = DatasetDict(dataset_dict)
        
        # Store metadata
        for split_name, split_dataset in self.dataset.items():
            split_dataset.label_to_id = train_dataset.label_to_id
            split_dataset.id_to_label = train_dataset.id_to_label
            split_dataset.num_labels = train_dataset.num_labels
        
        logger.info(f"Data preparation complete: {[f'{k}={len(v)}' for k, v in self.dataset.items()]}")
        
        return self.dataset
    
    def train(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        metrics_callback: Optional[Callable[[Dict[str, float]], None]] = None
    ) -> Dict[str, Any]:
        """Train the model using the prepared data."""
        
        if self.dataset is None:
            raise TrainingError(
                message="Data not prepared. Call prepare_data() first.",
                error_code=ErrorCodes.TRAINING_FAILED
            )
        
        self.progress_callback = progress_callback
        self.metrics_callback = metrics_callback
        
        start_time = time.time()
        
        try:
            logger.info(f"Starting training with {self.training_config.model_name}")
            
            # Create output directory
            output_dir = Path(self.training_config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize trainer based on configuration
            if self.training_config.use_distillation:
                result = self._train_with_distillation()
            else:
                result = self._train_standard()
            
            # Calculate total training time
            total_time = time.time() - start_time
            result['total_training_time'] = total_time
            
            # Save training history
            self._save_training_history(result)
            
            logger.info(f"Training completed in {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise TrainingError(
                message=f"Training failed: {str(e)}",
                error_code=ErrorCodes.TRAINING_FAILED,
                original_error=e
            )
    
    def _train_with_distillation(self) -> Dict[str, Any]:
        """Train using knowledge distillation."""
        
        logger.info("Training with knowledge distillation")
        
        # Create distillation configuration
        distillation_config = DistillationConfig(
            student_model_name=self.training_config.model_name,
            teacher_model_name=self.training_config.teacher_model,
            temperature=self.training_config.distillation_temperature,
            alpha_distillation=self.training_config.distillation_alpha,
            alpha_student=1.0 - self.training_config.distillation_alpha,
            learning_rate=self.training_config.learning_rate,
            num_epochs=self.training_config.num_epochs,
            batch_size=self.training_config.batch_size,
            warmup_ratio=self.training_config.warmup_ratio,
            weight_decay=self.training_config.weight_decay,
            early_stopping_patience=self.config.early_stopping_patience,  # Use config value, not hardcoded
            use_mixed_precision=self.training_config.use_mixed_precision,
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            metric_for_best_model=self.training_config.metric_for_best_model,
            output_dir=self.training_config.output_dir
        )
        
        # Initialize distillation trainer
        self.distillation_trainer = DistillationTrainer(
            config=self.config,
            distillation_config=distillation_config,
            model_manager=self.model_manager,
            validator=self.validator
        )
        
        # Prepare models
        train_dataset = self.dataset['train']
        self.distillation_trainer.prepare_models(
            num_labels=train_dataset.num_labels,
            label_to_id=train_dataset.label_to_id,
            id_to_label=train_dataset.id_to_label
        )
        
        # Train
        eval_dataset = self.dataset.get('validation')
        training_result = self.distillation_trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            progress_callback=self.progress_callback,
            metrics_callback=self.metrics_callback
        )
        
        # Add training type to result
        training_result['training_type'] = 'distillation'
        training_result['distillation_config'] = distillation_config.__dict__
        
        return training_result
    
    def _train_standard(self) -> Dict[str, Any]:
        """Train using standard fine-tuning."""
        
        logger.info("Training with standard fine-tuning")
        
        from transformers import (
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            EarlyStoppingCallback
        )
        
        # Load model
        train_dataset = self.dataset['train']
        model = AutoModelForSequenceClassification.from_pretrained(
            self.training_config.model_name,
            num_labels=train_dataset.num_labels,
            label2id=train_dataset.label_to_id,
            id2label=train_dataset.id_to_label
        )
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.training_config.batch_size,
            per_device_eval_batch_size=self.training_config.batch_size,
            learning_rate=self.training_config.learning_rate,
            warmup_ratio=self.training_config.warmup_ratio,
            weight_decay=self.training_config.weight_decay,
            evaluation_strategy=self.training_config.evaluation_strategy,
            save_strategy=self.training_config.save_strategy,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            load_best_model_at_end=True,
            save_total_limit=3,
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            bf16=self.training_config.use_mixed_precision == "bf16",
            fp16=self.training_config.use_mixed_precision == "fp16",
            logging_steps=10,
            report_to=None  # Disable wandb/tensorboard for now
        )
        
        # Create data collator
        data_collator = self.dataset_processor.create_data_collator(self.tokenizer)
        
        # Define compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = torch.argmax(torch.from_numpy(predictions), dim=-1)
            
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average='macro')
            precision = precision_score(labels, predictions, average='macro')
            recall = recall_score(labels, predictions, average='macro')
            
            return {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        # Create trainer
        eval_dataset = self.dataset.get('validation')
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        
        # Evaluate
        eval_result = {}
        if eval_dataset:
            eval_result = trainer.evaluate()
        
        return {
            'training_type': 'standard',
            'train_result': train_result,
            'eval_result': eval_result,
            'training_config': self.training_config.__dict__
        }
    
    def evaluate(
        self,
        test_dataset: Optional[Dataset] = None,
        include_llm_jury: bool = True
    ) -> EvaluationMetrics:
        """Comprehensive evaluation of the trained model."""
        
        if test_dataset is None:
            test_dataset = self.dataset.get('test')
        
        if test_dataset is None:
            raise TrainingError(
                message="No test dataset available for evaluation",
                error_code=ErrorCodes.EVALUATION_FAILED
            )
        
        logger.info("Starting comprehensive model evaluation")
        
        # Get model for evaluation
        if self.distillation_trainer is not None:
            model = self.distillation_trainer.student_model
            tokenizer = self.distillation_trainer.tokenizer
        else:
            # Load from output directory
            output_dir = Path(self.training_config.output_dir)
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(output_dir)
            tokenizer = AutoTokenizer.from_pretrained(output_dir)
        
        # Perform evaluation
        evaluation_result = self.evaluator.evaluate_model(
            model=model,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            include_llm_jury=include_llm_jury,
            llm_jury_model=self.model_manager.select_model(strategy="quality", task_type="evaluation")
        )
        
        logger.info(f"Evaluation completed - F1: {evaluation_result.f1_macro:.4f}")
        
        return evaluation_result
    
    def export_model(
        self,
        export_path: Optional[Union[str, Path]] = None,
        formats: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """Export trained model in various formats."""
        
        if export_path is None:
            export_path = Path(self.training_config.output_dir) / "exported"
        
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        if formats is None:
            formats = self.training_config.export_formats
        
        logger.info(f"Exporting model to {export_path} in formats: {formats}")
        
        exported_paths = {}
        
        # Get model and tokenizer
        if self.distillation_trainer is not None:
            model = self.distillation_trainer.student_model
            tokenizer = self.distillation_trainer.tokenizer
        else:
            # Load from output directory
            output_dir = Path(self.training_config.output_dir)
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(output_dir)
            tokenizer = AutoTokenizer.from_pretrained(output_dir)
        
        # Export in different formats
        for format_name in formats:
            format_path = export_path / format_name
            format_path.mkdir(parents=True, exist_ok=True)
            
            if format_name in ['transformers', 'huggingface']:
                # Standard HuggingFace format
                model.save_pretrained(format_path)
                tokenizer.save_pretrained(format_path)
                exported_paths[format_name] = format_path
                
            elif format_name == 'safetensors':
                # SafeTensors format
                model.save_pretrained(format_path, safe_serialization=True)
                tokenizer.save_pretrained(format_path)
                exported_paths[format_name] = format_path
                
            elif format_name == 'onnx' or self.training_config.export_onnx:
                # ONNX export
                try:
                    self._export_onnx(model, tokenizer, format_path)
                    exported_paths['onnx'] = format_path
                except Exception as e:
                    logger.warning(f"ONNX export failed: {e}")
            
            else:
                logger.warning(f"Unknown export format: {format_name}")
        
        # Save export metadata
        metadata = {
            'model_name': self.training_config.model_name,
            'task_name': self.training_config.task_name,
            'export_timestamp': time.time(),
            'formats': list(exported_paths.keys()),
            'training_config': self.training_config.__dict__
        }
        
        with open(export_path / "export_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model exported successfully: {list(exported_paths.keys())}")
        
        return exported_paths
    
    def _export_onnx(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        output_path: Path
    ) -> None:
        """Export model to ONNX format."""
        
        import torch.onnx
        
        model.eval()
        
        # Create dummy input
        dummy_input = {
            'input_ids': torch.randint(0, tokenizer.vocab_size, (1, self.training_config.max_sequence_length)),
            'attention_mask': torch.ones(1, self.training_config.max_sequence_length, dtype=torch.long)
        }
        
        # Export to ONNX
        torch.onnx.export(
            model,
            (dummy_input,),
            output_path / "model.onnx",
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size'}
            },
            opset_version=14
        )
        
        # Save tokenizer
        tokenizer.save_pretrained(output_path)
    
    def _save_training_history(self, training_result: Dict[str, Any]) -> None:
        """Save training history and results."""
        
        output_dir = Path(self.training_config.output_dir)
        
        # Save detailed training history
        history_file = output_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(training_result, f, indent=2, default=str)
        
        # Save configuration
        config_file = output_dir / "training_config.json"
        config_data = {
            'training_config': self.training_config.__dict__,
            'model_config': self.config.to_dict(),
            'training_timestamp': time.time()
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        logger.info(f"Training history saved to {output_dir}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        
        if self.distillation_trainer is not None and self.distillation_trainer.student_model is not None:
            model = self.distillation_trainer.student_model
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            return {
                'model_name': self.training_config.model_name,
                'task_name': self.training_config.task_name,
                'num_labels': model.num_labels,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
                'config': model.config.to_dict() if hasattr(model.config, 'to_dict') else str(model.config)
            }
        
        return {'status': 'No model loaded'}
    
    def load_model(
        self,
        model_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Load a previously trained model."""
        
        model_path = Path(model_path)
        
        # Load configuration if provided
        if config_path:
            config_path = Path(config_path)
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                
            # Update training config
            if 'training_config' in config_data:
                for key, value in config_data['training_config'].items():
                    if hasattr(self.training_config, key):
                        setattr(self.training_config, key, value)
        
        # Initialize distillation trainer if needed
        if self.training_config.use_distillation:
            distillation_config = DistillationConfig(
                student_model_name=self.training_config.model_name,
                output_dir=str(model_path.parent)
            )
            
            self.distillation_trainer = DistillationTrainer(
                config=self.config,
                distillation_config=distillation_config,
                model_manager=self.model_manager
            )
            
            self.distillation_trainer.load_model(model_path)
        
        logger.info(f"Model loaded from {model_path}")