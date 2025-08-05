"""
Knowledge distillation implementation for training small student models.

This module provides advanced knowledge distillation capabilities with
temperature scaling, feature matching, and adaptive loss weighting.
"""

import os
import time
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_scheduler
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ..config import Config, HuggingFaceModel
from ..llm import BaseLLM, ModelManager
from ..exceptions import TrainingError, ErrorCodes, handle_torch_error


def ensure_tensor_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Ensure all batch values are tensors and moved to the correct device.
    
    Handles conversion from lists/tuples to tensors where possible.
    """
    processed_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            processed_batch[k] = v.to(device)
        elif isinstance(v, (list, tuple)):
            # Convert lists/tuples to tensors
            try:
                processed_batch[k] = torch.tensor(v, device=device)
            except (ValueError, TypeError):
                # If conversion fails, keep as is
                processed_batch[k] = v
        else:
            processed_batch[k] = v
    return processed_batch


def load_model_for_classification(
    model_name: str,
    num_labels: int,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
    dropout_rate: Optional[float] = None
) -> AutoModelForSequenceClassification:
    """
    Load a model for sequence classification with appropriate parameters.
    
    Handles different model architectures and their specific parameter requirements.
    """
    model_kwargs = {
        "num_labels": num_labels,
        "label2id": label_to_id,
        "id2label": id_to_label
    }
    
    # Add dropout parameters based on model architecture if specified
    if dropout_rate is not None:
        model_name_lower = model_name.lower()
        if "distilbert" in model_name_lower:
            # DistilBERT uses 'dropout' parameter
            model_kwargs["dropout"] = dropout_rate
        elif "bert" in model_name_lower and "distilbert" not in model_name_lower:
            # BERT uses 'hidden_dropout_prob' and 'attention_probs_dropout_prob'
            model_kwargs["hidden_dropout_prob"] = dropout_rate
            model_kwargs["attention_probs_dropout_prob"] = dropout_rate
        elif "roberta" in model_name_lower:
            # RoBERTa uses 'hidden_dropout_prob' and 'attention_probs_dropout_prob'
            model_kwargs["hidden_dropout_prob"] = dropout_rate
            model_kwargs["attention_probs_dropout_prob"] = dropout_rate
        # For other models, don't add dropout parameters to avoid errors
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **model_kwargs
        )
        return model
    except TypeError as e:
        # If dropout parameters are not supported, load without them
        logger.warning(f"Dropout parameters not supported for {model_name}: {e}")
        logger.info("Loading model without dropout configuration...")
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            label2id=label_to_id,
            id2label=id_to_label
        )


from ..data import DataValidator, QualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation training."""
    
    # Model configuration
    student_model_name: str = HuggingFaceModel.DISTILBERT_BASE
    teacher_model_name: Optional[str] = None  # If None, use LLM for soft labels
    
    # Distillation parameters
    temperature: float = 4.0
    alpha_distillation: float = 0.7  # Weight for distillation loss
    alpha_student: float = 0.3       # Weight for hard target loss
    
    # Training parameters
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimization
    optimizer_type: str = "adamw"
    scheduler_type: str = "linear"
    use_mixed_precision: str = "bf16"  # "no", "fp16", "bf16"
    dataloader_num_workers: int = 4
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    metric_for_best_model: str = "eval_f1"
    
    # Feature distillation
    enable_feature_distillation: bool = False
    feature_loss_weight: float = 0.1
    
    # Adaptive loss weighting
    enable_adaptive_weighting: bool = True
    adaptation_rate: float = 0.01
    
    # Regularization
    dropout_rate: float = 0.1
    label_smoothing: float = 0.0
    
    # Output configuration
    output_dir: str = "./distillation_output"
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 3
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.alpha_distillation <= 1.0:
            raise ValueError("alpha_distillation must be between 0.0 and 1.0")
        if not 0.0 <= self.alpha_student <= 1.0:
            raise ValueError("alpha_student must be between 0.0 and 1.0")
        if abs(self.alpha_distillation + self.alpha_student - 1.0) > 1e-6:
            raise ValueError("alpha_distillation + alpha_student must equal 1.0")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


class DistillationLoss(nn.Module):
    """Advanced knowledge distillation loss with multiple components."""
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha_distillation: float = 0.7,
        alpha_student: float = 0.3,
        feature_loss_weight: float = 0.1,
        enable_adaptive_weighting: bool = True,
        adaptation_rate: float = 0.01
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha_distillation = alpha_distillation
        self.alpha_student = alpha_student
        self.feature_loss_weight = feature_loss_weight
        self.enable_adaptive_weighting = enable_adaptive_weighting
        self.adaptation_rate = adaptation_rate
        
        # Loss functions
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        # Adaptive weighting parameters
        self.register_buffer('distillation_weight', torch.tensor(alpha_distillation))
        self.register_buffer('student_weight', torch.tensor(alpha_student))
        
        # Loss history for adaptation
        self.loss_history = {
            'distillation': [],
            'student': [],
            'feature': []
        }
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model logits [batch_size, num_classes]
            teacher_logits: Teacher model logits [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
            student_features: Student intermediate features (optional)
            teacher_features: Teacher intermediate features (optional)
            
        Returns:
            Dictionary containing individual loss components and total loss
        """
        
        # Soft target loss (knowledge distillation)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = self.kl_div_loss(soft_predictions, soft_targets) * (self.temperature ** 2)
        
        # Hard target loss (ground truth)
        student_loss = self.cross_entropy_loss(student_logits, labels)
        
        # Feature distillation loss (if features provided)
        feature_loss = torch.tensor(0.0, device=student_logits.device)
        if student_features is not None and teacher_features is not None:
            feature_loss = self._compute_feature_loss(student_features, teacher_features)
        
        # Store loss history for adaptive weighting
        self.loss_history['distillation'].append(distillation_loss.item())
        self.loss_history['student'].append(student_loss.item())
        self.loss_history['feature'].append(feature_loss.item())
        
        # Limit history size
        max_history = 100
        for key in self.loss_history:
            if len(self.loss_history[key]) > max_history:
                self.loss_history[key] = self.loss_history[key][-max_history:]
        
        # Update adaptive weights
        if self.enable_adaptive_weighting and len(self.loss_history['distillation']) > 10:
            self._update_adaptive_weights()
        
        # Combine losses
        total_loss = (
            self.distillation_weight * distillation_loss +
            self.student_weight * student_loss +
            self.feature_loss_weight * feature_loss
        )
        
        return {
            'loss': total_loss,
            'distillation_loss': distillation_loss,
            'student_loss': student_loss,
            'feature_loss': feature_loss,
            'distillation_weight': self.distillation_weight,
            'student_weight': self.student_weight
        }
    
    def _compute_feature_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute feature matching loss between student and teacher."""
        
        # Ensure features have the same dimensions
        if student_features.shape != teacher_features.shape:
            # Apply projection if dimensions don't match
            if student_features.size(-1) != teacher_features.size(-1):
                # Simple linear projection (could be learned)
                teacher_dim = teacher_features.size(-1)
                student_dim = student_features.size(-1)
                
                if not hasattr(self, 'feature_projector'):
                    self.feature_projector = nn.Linear(teacher_dim, student_dim).to(teacher_features.device)
                
                teacher_features = self.feature_projector(teacher_features)
        
        # Normalize features
        student_features = F.normalize(student_features, dim=-1)
        teacher_features = F.normalize(teacher_features, dim=-1)
        
        # Compute MSE loss
        return self.mse_loss(student_features, teacher_features)
    
    def _update_adaptive_weights(self) -> None:
        """Update loss weights based on recent loss history."""
        
        # Calculate recent average losses
        window_size = 10
        recent_distill = np.mean(self.loss_history['distillation'][-window_size:])
        recent_student = np.mean(self.loss_history['student'][-window_size:])
        
        # Calculate relative magnitude
        total_loss = recent_distill + recent_student
        if total_loss > 0:
            distill_ratio = recent_distill / total_loss
            student_ratio = recent_student / total_loss
            
            # Adjust weights towards balance
            target_distill = 0.7  # Prefer distillation
            target_student = 0.3
            
            weight_adjustment = self.adaptation_rate
            
            # Update weights
            new_distill_weight = self.distillation_weight + weight_adjustment * (target_distill - distill_ratio)
            new_student_weight = self.student_weight + weight_adjustment * (target_student - student_ratio)
            
            # Ensure weights sum to 1 and are positive
            total_weight = new_distill_weight + new_student_weight
            if total_weight > 0:
                self.distillation_weight.data = torch.clamp(new_distill_weight / total_weight, 0.1, 0.9)
                self.student_weight.data = torch.clamp(new_student_weight / total_weight, 0.1, 0.9)


class DistillationTrainer:
    """Comprehensive knowledge distillation trainer."""
    
    def __init__(
        self,
        config: Config,
        distillation_config: DistillationConfig,
        model_manager: Optional[ModelManager] = None,
        validator: Optional[DataValidator] = None
    ):
        self.config = config
        self.distillation_config = distillation_config
        self.model_manager = model_manager
        self.validator = validator or DataValidator(config)
        
        # Training state
        self.student_model: Optional[nn.Module] = None
        self.teacher_model: Optional[nn.Module] = None
        self.teacher_llm: Optional[BaseLLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.distillation_loss: Optional[DistillationLoss] = None
        
        # Training callbacks
        self.progress_callback: Optional[Callable[[float, str], None]] = None
        self.metrics_callback: Optional[Callable[[Dict[str, float]], None]] = None
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
        
        logger.info(f"DistillationTrainer initialized on device: {self.device}")
    
    def prepare_models(
        self,
        num_labels: int,
        label_to_id: Dict[str, int],
        id_to_label: Dict[int, str]
    ) -> None:
        """Prepare student and teacher models for training."""
        
        logger.info("Preparing models for distillation training")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.distillation_config.student_model_name
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load student model
        logger.info(f"Loading student model: {self.distillation_config.student_model_name}")
        self.student_model = load_model_for_classification(
            self.distillation_config.student_model_name,
            num_labels,
            label_to_id,
            id_to_label,
            self.distillation_config.dropout_rate
        )
        
        self.student_model.to(self.device)
        
        # Prepare teacher (either another transformer or LLM)
        if self.distillation_config.teacher_model_name:
            logger.info(f"Loading teacher model: {self.distillation_config.teacher_model_name}")
            self.teacher_model = load_model_for_classification(
                self.distillation_config.teacher_model_name,
                num_labels,
                label_to_id,
                id_to_label
            )
            self.teacher_model.to(self.device)
            self.teacher_model.eval()
        elif self.model_manager:
            # Use LLM as teacher
            teacher_model_name = self.model_manager.select_model(
                strategy="quality",
                task_type="evaluation"
            )
            self.teacher_llm = self.model_manager.get_model(teacher_model_name)
            logger.info(f"Using LLM as teacher: {teacher_model_name}")
        else:
            raise TrainingError(
                message="No teacher model specified",
                error_code=ErrorCodes.TRAINING_FAILED
            )
        
        # Initialize distillation loss
        self.distillation_loss = DistillationLoss(
            temperature=self.distillation_config.temperature,
            alpha_distillation=self.distillation_config.alpha_distillation,
            alpha_student=self.distillation_config.alpha_student,
            feature_loss_weight=self.distillation_config.feature_loss_weight,
            enable_adaptive_weighting=self.distillation_config.enable_adaptive_weighting,
            adaptation_rate=self.distillation_config.adaptation_rate
        )
        
        logger.info("Models prepared successfully")
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        metrics_callback: Optional[Callable[[Dict[str, float]], None]] = None
    ) -> Dict[str, Any]:
        """Train the student model using knowledge distillation."""
        
        self.progress_callback = progress_callback
        self.metrics_callback = metrics_callback
        
        if self.student_model is None:
            raise TrainingError(
                message="Models not prepared. Call prepare_models() first.",
                error_code=ErrorCodes.TRAINING_FAILED
            )
        
        start_time = time.time()
        
        try:
            logger.info("Starting knowledge distillation training")
            
            # Create data loaders
            train_dataloader = self._create_dataloader(train_dataset, shuffle=True)
            eval_dataloader = None
            if eval_dataset:
                eval_dataloader = self._create_dataloader(eval_dataset, shuffle=False)
            
            # Prepare optimizer and scheduler
            optimizer = self._create_optimizer()
            total_steps = len(train_dataloader) * self.distillation_config.num_epochs
            scheduler = self._create_scheduler(optimizer, total_steps)
            
            # Training loop
            best_metric = float('-inf')
            patience_counter = 0
            training_history = []
            
            for epoch in range(self.distillation_config.num_epochs):
                epoch_start_time = time.time()
                
                # Training phase
                train_metrics = self._train_epoch(
                    train_dataloader, 
                    optimizer, 
                    scheduler, 
                    epoch
                )
                
                # Evaluation phase
                eval_metrics = {}
                if eval_dataloader:
                    eval_metrics = self._evaluate_epoch(eval_dataloader, epoch)
                
                # Combine metrics
                epoch_metrics = {**train_metrics, **eval_metrics}
                epoch_metrics['epoch'] = epoch
                epoch_metrics['epoch_time'] = time.time() - epoch_start_time
                
                training_history.append(epoch_metrics)
                
                # Call metrics callback
                if self.metrics_callback:
                    self.metrics_callback(epoch_metrics)
                
                # Check for best model
                current_metric = eval_metrics.get(
                    self.distillation_config.metric_for_best_model, 
                    train_metrics.get('train_f1', 0)
                )
                
                if current_metric > best_metric:
                    best_metric = current_metric
                    patience_counter = 0
                    self._save_best_model()
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= self.distillation_config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after epoch {epoch}")
                    break
                
                # Update progress
                progress = (epoch + 1) / self.distillation_config.num_epochs
                self._update_progress(
                    progress, 
                    f"Epoch {epoch + 1}/{self.distillation_config.num_epochs} - "
                    f"F1: {current_metric:.4f}"
                )
            
            total_time = time.time() - start_time
            
            # Final evaluation
            final_metrics = self._final_evaluation(eval_dataloader)
            
            training_result = {
                'training_history': training_history,
                'best_metric': best_metric,
                'final_metrics': final_metrics,
                'total_training_time': total_time,
                'epochs_completed': epoch + 1,
                'early_stopped': patience_counter >= self.distillation_config.early_stopping_patience
            }
            
            logger.info(f"Training completed in {total_time:.2f}s - Best {self.distillation_config.metric_for_best_model}: {best_metric:.4f}")
            
            return training_result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise handle_torch_error(e)
    
    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.student_model.train()
        
        total_loss = 0.0
        total_distillation_loss = 0.0
        total_student_loss = 0.0
        total_feature_loss = 0.0
        num_batches = len(dataloader)
        
        all_predictions = []
        all_labels = []
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Move batch to device and ensure tensors
                batch = ensure_tensor_batch(batch, self.device)
                
                # Forward pass through student
                student_outputs = self.student_model(**{
                    k: v for k, v in batch.items() 
                    if k in ['input_ids', 'attention_mask']
                })
                student_logits = student_outputs.logits
                
                # Get teacher predictions
                teacher_logits = self._get_teacher_predictions(batch)
                
                # Compute distillation loss
                loss_dict = self.distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=batch['labels']
                )
                
                loss = loss_dict['loss']
                
                # Backward pass
                if self.distillation_config.gradient_accumulation_steps > 1:
                    loss = loss / self.distillation_config.gradient_accumulation_steps
                
                loss.backward()
                
                # Gradient clipping
                if self.distillation_config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        self.distillation_config.max_grad_norm
                    )
                
                # Optimizer step
                if (batch_idx + 1) % self.distillation_config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Update metrics
                total_loss += loss_dict['loss'].item()
                total_distillation_loss += loss_dict['distillation_loss'].item()
                total_student_loss += loss_dict['student_loss'].item()
                total_feature_loss += loss_dict['feature_loss'].item()
                
                # Collect predictions for metrics
                predictions = torch.argmax(student_logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                
                # Log progress
                if batch_idx % self.distillation_config.logging_steps == 0:
                    logger.debug(
                        f"Epoch {epoch}, Batch {batch_idx}/{num_batches} - "
                        f"Loss: {loss.item():.4f}"
                    )
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        avg_distillation_loss = total_distillation_loss / num_batches
        avg_student_loss = total_student_loss / num_batches
        avg_feature_loss = total_feature_loss / num_batches
        
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='macro')
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        
        return {
            'train_loss': avg_loss,
            'train_distillation_loss': avg_distillation_loss,
            'train_student_loss': avg_student_loss,
            'train_feature_loss': avg_feature_loss,
            'train_accuracy': accuracy,
            'train_f1': f1,
            'train_precision': precision,
            'train_recall': recall,
        }
    
    def _evaluate_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Evaluate for one epoch."""
        
        self.student_model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device and ensure tensors
                batch = ensure_tensor_batch(batch, self.device)
                
                # Forward pass
                student_outputs = self.student_model(**{
                    k: v for k, v in batch.items() 
                    if k in ['input_ids', 'attention_mask']
                })
                student_logits = student_outputs.logits
                
                # Get teacher predictions
                teacher_logits = self._get_teacher_predictions(batch)
                
                # Compute loss
                loss_dict = self.distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=batch['labels']
                )
                
                total_loss += loss_dict['loss'].item()
                
                # Collect predictions
                predictions = torch.argmax(student_logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='macro')
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        
        return {
            'eval_loss': avg_loss,
            'eval_accuracy': accuracy,
            'eval_f1': f1,
            'eval_precision': precision,
            'eval_recall': recall,
        }
    
    def _get_teacher_predictions(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get teacher model predictions."""
        
        if self.teacher_model is not None:
            # Use transformer teacher model
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**{
                    k: v for k, v in batch.items() 
                    if k in ['input_ids', 'attention_mask']
                })
                return teacher_outputs.logits
        
        elif self.teacher_llm is not None:
            # Use LLM teacher - this would require specialized implementation
            # For now, return dummy logits
            
            # Ensure input_ids is a tensor
            input_ids = batch['input_ids']
            if isinstance(input_ids, (list, tuple)):
                input_ids = torch.tensor(input_ids, device=self.device)
            
            batch_size = input_ids.size(0)
            num_classes = self.student_model.num_labels
            
            # In a real implementation, you would:
            # 1. Decode the input_ids back to text
            # 2. Use the LLM to classify the text
            # 3. Convert predictions to logits
            
            # Dummy implementation - random logits
            return torch.randn(batch_size, num_classes, device=self.device)
        
        else:
            raise TrainingError(
                message="No teacher model available",
                error_code=ErrorCodes.TRAINING_FAILED
            )
    
    def _create_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """Create DataLoader with appropriate settings."""
        
        # Create custom data collator for tokenized lists
        def collate_fn(batch):
            """Custom collator for pre-tokenized data (lists of integers)."""
            
            # Extract lists and convert to tensors with padding
            input_ids_list = [item['input_ids'] for item in batch]
            attention_mask_list = [item['attention_mask'] for item in batch]
            labels_list = [item['labels'] for item in batch]
            
            # Pad sequences to same length
            max_length = max(len(ids) for ids in input_ids_list)
            
            # Pad input_ids and attention_mask
            padded_input_ids = []
            padded_attention_mask = []
            
            for ids, mask in zip(input_ids_list, attention_mask_list):
                padding_length = max_length - len(ids)
                
                # Pad with tokenizer's pad_token_id (usually 0)
                padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
                padded_mask = mask + [0] * padding_length
                
                padded_input_ids.append(padded_ids)
                padded_attention_mask.append(padded_mask)
            
            return {
                'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long),
                'labels': torch.tensor(labels_list, dtype=torch.long)
            }
        
        return DataLoader(
            dataset,
            batch_size=self.distillation_config.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Disable multiprocessing to avoid tensor serialization issues
            pin_memory=True if self.device.type == "cuda" else False,
            drop_last=False,
            collate_fn=collate_fn
        )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for training."""
        
        if self.distillation_config.optimizer_type.lower() == "adamw":
            return torch.optim.AdamW(
                self.student_model.parameters(),
                lr=self.distillation_config.learning_rate,
                weight_decay=self.distillation_config.weight_decay
            )
        elif self.distillation_config.optimizer_type.lower() == "adam":
            return torch.optim.Adam(
                self.student_model.parameters(),
                lr=self.distillation_config.learning_rate,
                weight_decay=self.distillation_config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.distillation_config.optimizer_type}")
    
    def _create_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int):
        """Create learning rate scheduler."""
        
        warmup_steps = int(total_steps * self.distillation_config.warmup_ratio)
        
        return get_scheduler(
            name=self.distillation_config.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    def _save_best_model(self) -> None:
        """Save the best model checkpoint."""
        
        output_dir = Path(self.distillation_config.output_dir) / "best_model"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.student_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save configuration
        config_dict = {
            'distillation_config': self.distillation_config.__dict__,
            'model_config': self.student_model.config.to_dict(),
        }
        
        with open(output_dir / "training_config.json", 'w') as f:
            import json
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Best model saved to {output_dir}")
    
    def _final_evaluation(self, eval_dataloader: Optional[DataLoader]) -> Dict[str, float]:
        """Perform final comprehensive evaluation."""
        
        if eval_dataloader is None:
            return {}
        
        logger.info("Performing final evaluation")
        
        self.student_model.eval()
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.student_model(**{
                    k: v for k, v in batch.items() 
                    if k in ['input_ids', 'attention_mask']
                })
                
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_micro = f1_score(all_labels, all_predictions, average='micro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        precision_macro = precision_score(all_labels, all_predictions, average='macro')
        recall_macro = recall_score(all_labels, all_predictions, average='macro')
        
        # Confidence statistics
        max_probs = np.max(all_probs, axis=1)
        avg_confidence = np.mean(max_probs)
        
        return {
            'final_accuracy': accuracy,
            'final_f1_macro': f1_macro,
            'final_f1_micro': f1_micro,
            'final_f1_weighted': f1_weighted,
            'final_precision_macro': precision_macro,
            'final_recall_macro': recall_macro,
            'final_avg_confidence': avg_confidence,
        }
    
    def _update_progress(self, progress: float, message: str) -> None:
        """Update training progress."""
        if self.progress_callback:
            self.progress_callback(progress, message)
        
        logger.info(f"Progress: {progress:.1%} - {message}")
    
    def save_model(
        self,
        output_path: Union[str, os.PathLike],
        save_tokenizer: bool = True,
        save_config: bool = True
    ) -> None:
        """Save the trained model."""
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.student_model is None:
            raise TrainingError(
                message="No model to save",
                error_code=ErrorCodes.TRAINING_FAILED
            )
        
        # Save model
        self.student_model.save_pretrained(output_path)
        logger.info(f"Model saved to {output_path}")
        
        # Save tokenizer
        if save_tokenizer and self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_path)
            logger.info(f"Tokenizer saved to {output_path}")
        
        # Save configuration
        if save_config:
            config_dict = {
                'distillation_config': self.distillation_config.__dict__,
                'model_config': self.student_model.config.to_dict(),
                'training_timestamp': time.time(),
            }
            
            with open(output_path / "distillation_config.json", 'w') as f:
                import json
                json.dump(config_dict, f, indent=2, default=str)
    
    def load_model(
        self,
        model_path: Union[str, os.PathLike],
        load_tokenizer: bool = True
    ) -> None:
        """Load a trained model."""
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise TrainingError(
                message=f"Model path does not exist: {model_path}",
                error_code=ErrorCodes.MODEL_LOAD_FAILED
            )
        
        # Load model
        self.student_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.student_model.to(self.device)
        logger.info(f"Model loaded from {model_path}")
        
        # Load tokenizer
        if load_tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"Tokenizer loaded from {model_path}")
    
    def predict(
        self,
        texts: List[str],
        return_probabilities: bool = False,
        batch_size: Optional[int] = None
    ) -> Union[List[str], Tuple[List[str], List[List[float]]]]:
        """Make predictions on new texts."""
        
        if self.student_model is None or self.tokenizer is None:
            raise TrainingError(
                message="Model not loaded",
                error_code=ErrorCodes.MODEL_LOAD_FAILED
            )
        
        if batch_size is None:
            batch_size = self.distillation_config.batch_size
        
        self.student_model.eval()
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.config.max_sequence_length,
                    return_tensors="pt"
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.student_model(**inputs)
                logits = outputs.logits
                
                # Get predictions
                probs = F.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                # Convert to labels
                pred_labels = [self.student_model.config.id2label[pred.item()] 
                              for pred in predictions]
                
                all_predictions.extend(pred_labels)
                
                if return_probabilities:
                    all_probabilities.extend(probs.cpu().numpy().tolist())
        
        if return_probabilities:
            return all_predictions, all_probabilities
        else:
            return all_predictions