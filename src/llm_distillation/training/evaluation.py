"""
Model evaluation and metrics calculation module.

This module provides comprehensive evaluation capabilities including
standard metrics, LLM-based evaluation, and detailed analysis.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn.functional as F
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from ..config import Config
from ..llm import BaseLLM, ModelManager, GenerationConfig
from ..exceptions import EvaluationError, ErrorCodes

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    
    # Basic classification metrics
    accuracy: float = 0.0
    precision_macro: float = 0.0
    precision_micro: float = 0.0
    precision_weighted: float = 0.0
    recall_macro: float = 0.0
    recall_micro: float = 0.0
    recall_weighted: float = 0.0
    f1_macro: float = 0.0
    f1_micro: float = 0.0
    f1_weighted: float = 0.0
    
    # Per-class metrics
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    per_class_support: Dict[str, int] = field(default_factory=dict)
    
    # Advanced metrics
    auc_score: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    
    # Confidence and calibration
    average_confidence: float = 0.0
    calibration_error: float = 0.0
    overconfidence_ratio: float = 0.0
    
    # LLM jury evaluation
    llm_jury_agreement: Optional[float] = None
    llm_jury_scores: Dict[str, float] = field(default_factory=dict)
    
    # Efficiency metrics
    inference_time_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    model_size_mb: float = 0.0
    
    # Additional metadata
    num_samples: int = 0
    num_classes: int = 0
    evaluation_timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = value
        return result
    
    def summary(self) -> str:
        """Get a summary string of key metrics."""
        return (
            f"Accuracy: {self.accuracy:.4f}, "
            f"F1-Macro: {self.f1_macro:.4f}, "
            f"F1-Micro: {self.f1_micro:.4f}, "
            f"Precision: {self.precision_macro:.4f}, "
            f"Recall: {self.recall_macro:.4f}"
        )


class ModelEvaluator:
    """Comprehensive model evaluation system."""
    
    def __init__(
        self,
        config: Config,
        model_manager: Optional[ModelManager] = None
    ):
        self.config = config
        self.model_manager = model_manager
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
        
        logger.info(f"ModelEvaluator initialized on device: {self.device}")
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        test_dataset: Dataset,
        include_llm_jury: bool = False,
        llm_jury_model: Optional[str] = None,
        llm_jury_sample_size: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> EvaluationMetrics:
        """Perform comprehensive model evaluation."""
        
        if batch_size is None:
            batch_size = self.config.batch_size
        
        if llm_jury_sample_size is None:
            llm_jury_sample_size = self.config.jury_sample_size
        
        logger.info(f"Starting evaluation on {len(test_dataset)} samples")
        
        start_time = time.time()
        
        # Basic evaluation
        predictions, probabilities, labels, inference_times = self._get_predictions(
            model, tokenizer, test_dataset, batch_size
        )
        
        # Calculate standard metrics
        metrics = self._calculate_standard_metrics(predictions, probabilities, labels)
        
        # Add efficiency metrics
        total_inference_time = sum(inference_times)
        metrics.inference_time_ms = total_inference_time * 1000 / len(test_dataset)
        metrics.throughput_samples_per_sec = len(test_dataset) / total_inference_time
        metrics.model_size_mb = self._calculate_model_size(model)
        
        # Calculate confidence and calibration metrics
        self._calculate_confidence_metrics(metrics, predictions, probabilities, labels)
        
        # LLM jury evaluation
        if include_llm_jury and self.model_manager:
            try:
                logger.info("Starting LLM jury evaluation")
                self._llm_jury_evaluation(
                    metrics, 
                    test_dataset, 
                    predictions, 
                    llm_jury_model,
                    llm_jury_sample_size
                )
            except Exception as e:
                logger.warning(f"LLM jury evaluation failed: {e}")
        
        metrics.num_samples = len(test_dataset)
        metrics.evaluation_timestamp = time.time()
        
        evaluation_time = time.time() - start_time
        logger.info(f"Evaluation completed in {evaluation_time:.2f}s - {metrics.summary()}")
        
        return metrics
    
    def _get_predictions(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        batch_size: int
    ) -> Tuple[List[int], List[List[float]], List[int], List[float]]:
        """Get model predictions for the dataset."""
        
        model.eval()
        model.to(self.device)
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        inference_times = []
        
        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch_start = time.time()
                
                # Get batch
                batch_end = min(i + batch_size, len(dataset))
                batch_data = dataset[i:batch_end]
                
                # Prepare inputs with dynamic padding
                input_ids_list = batch_data['input_ids']
                attention_mask_list = batch_data['attention_mask']
                labels_list = batch_data['labels']
                
                # Pad sequences to same length
                max_length = max(len(ids) for ids in input_ids_list)
                
                # Pad input_ids and attention_mask
                padded_input_ids = []
                padded_attention_mask = []
                
                for ids, mask in zip(input_ids_list, attention_mask_list):
                    padding_length = max_length - len(ids)
                    
                    # Pad with tokenizer's pad_token_id (usually 0)
                    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                    padded_ids = ids + [pad_token_id] * padding_length
                    padded_mask = mask + [0] * padding_length
                    
                    padded_input_ids.append(padded_ids)
                    padded_attention_mask.append(padded_mask)
                
                input_ids = torch.tensor(padded_input_ids, dtype=torch.long).to(self.device)
                attention_mask = torch.tensor(padded_attention_mask, dtype=torch.long).to(self.device)
                labels = torch.tensor(labels_list, dtype=torch.long).to(self.device)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Calculate probabilities and predictions
                probabilities = F.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Record inference time
                batch_time = time.time() - batch_start
                inference_times.extend([batch_time / len(batch_data['labels'])] * len(batch_data['labels']))
        
        return all_predictions, all_probabilities, all_labels, inference_times
    
    def _calculate_standard_metrics(
        self,
        predictions: List[int],
        probabilities: List[List[float]],
        labels: List[int]
    ) -> EvaluationMetrics:
        """Calculate standard classification metrics."""
        
        metrics = EvaluationMetrics()
        
        # Basic metrics
        metrics.accuracy = accuracy_score(labels, predictions)
        
        # Precision, Recall, F1 with different averaging
        metrics.precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
        metrics.precision_micro = precision_score(labels, predictions, average='micro', zero_division=0)
        metrics.precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)
        
        metrics.recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
        metrics.recall_micro = recall_score(labels, predictions, average='micro', zero_division=0)
        metrics.recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)
        
        metrics.f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        metrics.f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
        metrics.f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        unique_labels = sorted(set(labels))
        metrics.num_classes = len(unique_labels)
        
        per_class_precision = precision_score(labels, predictions, average=None, zero_division=0)
        per_class_recall = recall_score(labels, predictions, average=None, zero_division=0)
        per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
        
        # Support (number of samples per class)
        from collections import Counter
        label_counts = Counter(labels)
        
        for i, label_idx in enumerate(unique_labels):
            label_name = str(label_idx)
            metrics.per_class_precision[label_name] = per_class_precision[i]
            metrics.per_class_recall[label_name] = per_class_recall[i]
            metrics.per_class_f1[label_name] = per_class_f1[i]
            metrics.per_class_support[label_name] = label_counts[label_idx]
        
        # Confusion matrix
        metrics.confusion_matrix = confusion_matrix(labels, predictions)
        
        # AUC score (for binary classification or multiclass with probability)
        try:
            if len(unique_labels) == 2:
                # Binary classification
                probs_positive = [prob[1] for prob in probabilities]
                metrics.auc_score = roc_auc_score(labels, probs_positive)
            elif len(unique_labels) > 2:
                # Multiclass - use one-vs-rest
                metrics.auc_score = roc_auc_score(labels, probabilities, multi_class='ovr', average='macro')
        except Exception as e:
            logger.warning(f"Could not calculate AUC score: {e}")
            metrics.auc_score = None
        
        return metrics
    
    def _calculate_confidence_metrics(
        self,
        metrics: EvaluationMetrics,
        predictions: List[int],
        probabilities: List[List[float]],
        labels: List[int]
    ) -> None:
        """Calculate confidence and calibration metrics."""
        
        # Average confidence (max probability)
        max_probs = [max(prob) for prob in probabilities]
        metrics.average_confidence = np.mean(max_probs)
        
        # Calibration error (Expected Calibration Error - ECE)
        metrics.calibration_error = self._calculate_ece(predictions, probabilities, labels)
        
        # Overconfidence ratio (samples with high confidence but wrong prediction)
        correct_predictions = [pred == label for pred, label in zip(predictions, labels)]
        high_confidence_threshold = 0.8
        high_confidence_wrong = [
            max_prob > high_confidence_threshold and not correct
            for max_prob, correct in zip(max_probs, correct_predictions)
        ]
        
        if max_probs:
            metrics.overconfidence_ratio = sum(high_confidence_wrong) / len(max_probs)
    
    def _calculate_ece(
        self,
        predictions: List[int],
        probabilities: List[List[float]],
        labels: List[int],
        n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error."""
        
        max_probs = np.array([max(prob) for prob in probabilities])
        predicted_labels = np.array(predictions)
        true_labels = np.array(labels)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Get samples in this confidence bin
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Calculate accuracy and confidence for this bin
                accuracy_in_bin = (predicted_labels[in_bin] == true_labels[in_bin]).mean()
                avg_confidence_in_bin = max_probs[in_bin].mean()
                
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    def _calculate_model_size(self, model: torch.nn.Module) -> float:
        """Calculate model size in MB."""
        
        total_params = sum(p.numel() for p in model.parameters())
        # Assuming float32 parameters (4 bytes each)
        size_mb = (total_params * 4) / (1024 * 1024)
        return size_mb
    
    def _llm_jury_evaluation(
        self,
        metrics: EvaluationMetrics,
        dataset: Dataset,
        predictions: List[int],
        llm_jury_model: Optional[str],
        sample_size: int
    ) -> None:
        """Perform LLM jury evaluation."""
        
        if not self.model_manager:
            logger.warning("No model manager available for LLM jury evaluation")
            return
        
        # Select jury model
        if llm_jury_model is None:
            llm_jury_model = self.model_manager.select_model(
                strategy="quality",
                task_type="evaluation"
            )
        
        jury_llm = self.model_manager.get_model(llm_jury_model)
        
        # Sample data for evaluation
        indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)
        
        agreements = []
        quality_scores = []
        
        for idx in indices:
            sample = dataset[int(idx)]
            text = sample.get('texts', [''])[0] if 'texts' in sample else ''
            true_label = sample.get('original_labels', [''])[0] if 'original_labels' in sample else ''
            predicted_label = predictions[idx]
            
            if not text or not true_label:
                continue
            
            # Get jury evaluation
            jury_result = self._get_jury_evaluation(
                jury_llm, text, true_label, predicted_label
            )
            
            if jury_result:
                agreements.append(jury_result['agreement'])
                quality_scores.append(jury_result['quality_score'])
        
        # Calculate jury metrics
        if agreements:
            metrics.llm_jury_agreement = np.mean(agreements)
            metrics.llm_jury_scores = {
                'agreement_rate': np.mean(agreements),
                'quality_score': np.mean(quality_scores),
                'samples_evaluated': len(agreements)
            }
    
    def _get_jury_evaluation(
        self,
        jury_llm: BaseLLM,
        text: str,
        true_label: str,
        predicted_label: str
    ) -> Optional[Dict[str, Any]]:
        """Get evaluation from LLM jury."""
        
        prompt = f"""Evaluate this text classification prediction:

Text: "{text}"
True Label: {true_label}
Predicted Label: {predicted_label}

Please evaluate:
1. Is the prediction correct? (yes/no)
2. Rate the prediction quality on a scale of 1-5:
   - 5: Perfect prediction
   - 4: Good prediction with minor issues
   - 3: Acceptable prediction
   - 2: Poor prediction but understandable
   - 1: Very poor prediction

Respond in JSON format:
{{"correct": true/false, "quality_score": 1-5, "explanation": "brief explanation"}}"""
        
        try:
            config = GenerationConfig(
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            response = jury_llm.generate_text(prompt, config)
            
            # Parse response
            import json
            result = json.loads(response.text)
            
            return {
                'agreement': result.get('correct', False),
                'quality_score': result.get('quality_score', 3),
                'explanation': result.get('explanation', '')
            }
            
        except Exception as e:
            logger.warning(f"Jury evaluation failed for sample: {e}")
            return None
    
    def create_evaluation_report(
        self,
        metrics: EvaluationMetrics,
        output_path: Optional[str] = None,
        include_plots: bool = True
    ) -> str:
        """Create a comprehensive evaluation report."""
        
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Evaluation Date: {time.ctime(metrics.evaluation_timestamp)}")
        report_lines.append(f"Number of Samples: {metrics.num_samples}")
        report_lines.append(f"Number of Classes: {metrics.num_classes}")
        report_lines.append("")
        
        # Overall Performance
        report_lines.append("OVERALL PERFORMANCE")
        report_lines.append("-" * 40)
        report_lines.append(f"Accuracy:           {metrics.accuracy:.4f}")
        report_lines.append(f"F1-Score (Macro):   {metrics.f1_macro:.4f}")
        report_lines.append(f"F1-Score (Micro):   {metrics.f1_micro:.4f}")
        report_lines.append(f"F1-Score (Weighted): {metrics.f1_weighted:.4f}")
        report_lines.append(f"Precision (Macro):  {metrics.precision_macro:.4f}")
        report_lines.append(f"Recall (Macro):     {metrics.recall_macro:.4f}")
        if metrics.auc_score is not None:
            report_lines.append(f"AUC Score:          {metrics.auc_score:.4f}")
        report_lines.append("")
        
        # Per-Class Performance
        report_lines.append("PER-CLASS PERFORMANCE")
        report_lines.append("-" * 40)
        report_lines.append(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
        report_lines.append("-" * 60)
        
        for class_name in sorted(metrics.per_class_precision.keys()):
            precision = metrics.per_class_precision[class_name]
            recall = metrics.per_class_recall[class_name]
            f1 = metrics.per_class_f1[class_name]
            support = metrics.per_class_support[class_name]
            
            report_lines.append(
                f"{class_name:<15} {precision:<10.4f} {recall:<10.4f} "
                f"{f1:<10.4f} {support:<10}"
            )
        report_lines.append("")
        
        # Confidence and Calibration
        report_lines.append("CONFIDENCE AND CALIBRATION")
        report_lines.append("-" * 40)
        report_lines.append(f"Average Confidence:     {metrics.average_confidence:.4f}")
        report_lines.append(f"Calibration Error (ECE): {metrics.calibration_error:.4f}")
        report_lines.append(f"Overconfidence Ratio:   {metrics.overconfidence_ratio:.4f}")
        report_lines.append("")
        
        # Efficiency Metrics
        report_lines.append("EFFICIENCY METRICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Model Size:             {metrics.model_size_mb:.2f} MB")
        report_lines.append(f"Inference Time:         {metrics.inference_time_ms:.2f} ms/sample")
        report_lines.append(f"Throughput:             {metrics.throughput_samples_per_sec:.2f} samples/sec")
        report_lines.append("")
        
        # LLM Jury Results
        if metrics.llm_jury_agreement is not None:
            report_lines.append("LLM JURY EVALUATION")
            report_lines.append("-" * 40)
            report_lines.append(f"Agreement Rate:         {metrics.llm_jury_agreement:.4f}")
            for metric_name, score in metrics.llm_jury_scores.items():
                report_lines.append(f"{metric_name.title():<20}: {score}")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save report
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            
            logger.info(f"Evaluation report saved to {output_path}")
            
            # Create plots if requested
            if include_plots:
                self._create_evaluation_plots(metrics, output_path)
        
        return report_text
    
    def _create_evaluation_plots(
        self,
        metrics: EvaluationMetrics,
        output_path: str
    ) -> None:
        """Create evaluation plots."""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from pathlib import Path
            
            output_dir = Path(output_path).parent
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Confusion Matrix
            if metrics.confusion_matrix is not None:
                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    metrics.confusion_matrix,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=range(metrics.num_classes),
                    yticklabels=range(metrics.num_classes)
                )
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 2. Per-Class Performance
            if metrics.per_class_f1:
                class_names = list(metrics.per_class_f1.keys())
                f1_scores = list(metrics.per_class_f1.values())
                precision_scores = [metrics.per_class_precision[name] for name in class_names]
                recall_scores = [metrics.per_class_recall[name] for name in class_names]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(class_names))
                width = 0.25
                
                ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
                ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
                ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
                
                ax.set_xlabel('Classes')
                ax.set_ylabel('Score')
                ax.set_title('Per-Class Performance Metrics')
                ax.set_xticks(x)
                ax.set_xticklabels(class_names, rotation=45)
                ax.legend()
                ax.set_ylim(0, 1)
                
                plt.tight_layout()
                plt.savefig(output_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. Metrics Summary
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Overall metrics
            overall_metrics = {
                'Accuracy': metrics.accuracy,
                'F1-Macro': metrics.f1_macro,
                'Precision': metrics.precision_macro,
                'Recall': metrics.recall_macro
            }
            
            ax1.bar(overall_metrics.keys(), overall_metrics.values(), alpha=0.7)
            ax1.set_title('Overall Performance Metrics')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1)
            
            # Confidence metrics
            confidence_metrics = {
                'Avg Confidence': metrics.average_confidence,
                'Calibration Error': metrics.calibration_error,
                'Overconfidence': metrics.overconfidence_ratio
            }
            
            ax2.bar(confidence_metrics.keys(), confidence_metrics.values(), alpha=0.7)
            ax2.set_title('Confidence & Calibration')
            ax2.set_ylabel('Score')
            ax2.tick_params(axis='x', rotation=45)
            
            # Efficiency metrics (normalized)
            ax3.text(0.1, 0.8, f'Model Size: {metrics.model_size_mb:.1f} MB', transform=ax3.transAxes)
            ax3.text(0.1, 0.6, f'Inference: {metrics.inference_time_ms:.1f} ms', transform=ax3.transAxes)
            ax3.text(0.1, 0.4, f'Throughput: {metrics.throughput_samples_per_sec:.1f} samples/s', transform=ax3.transAxes)
            ax3.set_title('Efficiency Metrics')
            ax3.axis('off')
            
            # LLM Jury (if available)
            if metrics.llm_jury_agreement is not None:
                jury_data = {
                    'Agreement Rate': metrics.llm_jury_agreement,
                    'Quality Score': metrics.llm_jury_scores.get('quality_score', 0) / 5  # Normalize to 0-1
                }
                ax4.bar(jury_data.keys(), jury_data.values(), alpha=0.7)
                ax4.set_title('LLM Jury Evaluation')
                ax4.set_ylabel('Score')
                ax4.set_ylim(0, 1)
            else:
                ax4.text(0.5, 0.5, 'No LLM Jury\nEvaluation', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('LLM Jury Evaluation')
                ax4.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'evaluation_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Evaluation plots saved to {output_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to create evaluation plots: {e}")
    
    def compare_models(
        self,
        model_metrics: Dict[str, EvaluationMetrics],
        output_path: Optional[str] = None
    ) -> str:
        """Compare multiple models and create comparison report."""
        
        if len(model_metrics) < 2:
            raise EvaluationError(
                message="Need at least 2 models for comparison",
                error_code=ErrorCodes.EVALUATION_FAILED
            )
        
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("MODEL COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Comparison Date: {time.ctime()}")
        report_lines.append(f"Models Compared: {', '.join(model_metrics.keys())}")
        report_lines.append("")
        
        # Performance Comparison
        report_lines.append("PERFORMANCE COMPARISON")
        report_lines.append("-" * 60)
        
        # Create comparison table
        metrics_to_compare = [
            ('Accuracy', 'accuracy'),
            ('F1-Macro', 'f1_macro'),
            ('Precision', 'precision_macro'),
            ('Recall', 'recall_macro'),
            ('Inference Time (ms)', 'inference_time_ms'),
            ('Model Size (MB)', 'model_size_mb')
        ]
        
        # Header
        header = f"{'Metric':<20}"
        for model_name in model_metrics.keys():
            header += f"{model_name:<15}"
        header += f"{'Best':<15}"
        report_lines.append(header)
        report_lines.append("-" * len(header))
        
        # Metrics rows
        for metric_display, metric_attr in metrics_to_compare:
            row = f"{metric_display:<20}"
            values = {}
            
            for model_name, metrics in model_metrics.items():
                value = getattr(metrics, metric_attr, 0)
                values[model_name] = value
                
                if 'time' in metric_attr.lower() or 'size' in metric_attr.lower():
                    # Lower is better for time and size
                    row += f"{value:<15.3f}"
                else:
                    # Higher is better for performance metrics
                    row += f"{value:<15.4f}"
            
            # Find best model for this metric
            if 'time' in metric_attr.lower() or 'size' in metric_attr.lower():
                best_model = min(values.keys(), key=lambda k: values[k])
            else:
                best_model = max(values.keys(), key=lambda k: values[k])
            
            row += f"{best_model:<15}"
            report_lines.append(row)
        
        report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        # Find overall best model (simple scoring)
        model_scores = {}
        for model_name, metrics in model_metrics.items():
            score = (
                metrics.f1_macro * 0.4 +
                metrics.accuracy * 0.3 +
                metrics.precision_macro * 0.15 +
                metrics.recall_macro * 0.15
            )
            model_scores[model_name] = score
        
        best_overall = max(model_scores.keys(), key=lambda k: model_scores[k])
        
        report_lines.append(f"Best Overall Performance: {best_overall}")
        report_lines.append(f"  Score: {model_scores[best_overall]:.4f}")
        report_lines.append("")
        
        # Efficiency recommendations
        efficiency_scores = {}
        for model_name, metrics in model_metrics.items():
            # Lower time and size is better
            eff_score = 1.0 / (metrics.inference_time_ms + 1) + 1.0 / (metrics.model_size_mb + 1)
            efficiency_scores[model_name] = eff_score
        
        most_efficient = max(efficiency_scores.keys(), key=lambda k: efficiency_scores[k])
        report_lines.append(f"Most Efficient Model: {most_efficient}")
        
        report_text = "\n".join(report_lines)
        
        # Save report
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Model comparison report saved to {output_path}")
        
        return report_text