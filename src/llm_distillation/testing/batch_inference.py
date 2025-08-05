"""
Batch inference management for processing multiple samples efficiently.

This module provides utilities for managing large-scale inference tasks,
including file processing, result export, and performance monitoring.
"""

import os
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import logging
from datetime import datetime

from .inference import ModelTester, InferenceResult, InferenceConfig

logger = logging.getLogger(__name__)


class BatchInferenceManager:
    """Manager for batch inference operations."""
    
    def __init__(
        self,
        model_tester: ModelTester,
        output_dir: Union[str, Path] = "inference_results"
    ):
        self.model_tester = model_tester
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"BatchInferenceManager initialized with output dir: {self.output_dir}")
    
    def process_text_file(
        self,
        input_file: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[InferenceResult]:
        """Process a text file with one sample per line."""
        input_file = Path(input_file)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_file}")
        
        # Read texts from file
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Processing {len(texts)} texts from {input_file}")
        
        # Perform batch inference
        results = self.model_tester.predict_batch(texts, progress_callback)
        
        # Save results if output file specified
        if output_file:
            self.save_results(results, output_file)
        
        return results
    
    def process_csv_file(
        self,
        input_file: Union[str, Path],
        text_column: str = "text",
        label_column: Optional[str] = None,
        output_file: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[InferenceResult]:
        """Process a CSV file with text data."""
        input_file = Path(input_file)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_file}")
        
        # Read CSV data
        texts = []
        true_labels = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if text_column in row and row[text_column].strip():
                    texts.append(row[text_column].strip())
                    if label_column and label_column in row:
                        true_labels.append(row[label_column].strip())
        
        logger.info(f"Processing {len(texts)} texts from CSV file {input_file}")
        
        # Perform batch inference
        results = self.model_tester.predict_batch(texts, progress_callback)
        
        # Add true labels if available
        if true_labels and len(true_labels) == len(results):
            for result, true_label in zip(results, true_labels):
                result.true_label = true_label
        
        # Save results if output file specified
        if output_file:
            self.save_results(results, output_file)
        
        return results
    
    def process_jsonl_file(
        self,
        input_file: Union[str, Path],
        text_field: str = "text",
        label_field: Optional[str] = "label",
        output_file: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[InferenceResult]:
        """Process a JSONL file with text data."""
        input_file = Path(input_file)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_file}")
        
        # Read JSONL data
        texts = []
        true_labels = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if text_field in data and data[text_field].strip():
                        texts.append(data[text_field].strip())
                        if label_field and label_field in data:
                            true_labels.append(data[label_field])
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    continue
        
        logger.info(f"Processing {len(texts)} texts from JSONL file {input_file}")
        
        # Perform batch inference
        results = self.model_tester.predict_batch(texts, progress_callback)
        
        # Add true labels if available
        if true_labels and len(true_labels) == len(results):
            for result, true_label in zip(results, true_labels):
                result.true_label = true_label
        
        # Save results if output file specified
        if output_file:
            self.save_results(results, output_file)
        
        return results
    
    def save_results(
        self,
        results: List[InferenceResult],
        output_file: Union[str, Path],
        format: str = "auto"
    ) -> None:
        """Save inference results to file."""
        output_file = Path(output_file)
        
        # Determine format from extension if auto
        if format == "auto":
            format = output_file.suffix.lower().lstrip('.')
            if format not in ['json', 'jsonl', 'csv']:
                format = 'jsonl'  # Default to JSONL
        
        logger.info(f"Saving {len(results)} results to {output_file} in {format} format")
        
        if format == 'json':
            self._save_json(results, output_file)
        elif format == 'jsonl':
            self._save_jsonl(results, output_file)
        elif format == 'csv':
            self._save_csv(results, output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_json(self, results: List[InferenceResult], output_file: Path) -> None:
        """Save results as JSON."""
        data = {
            'metadata': {
                'num_samples': len(results),
                'model_name': results[0].model_name if results else 'Unknown',
                'model_path': results[0].model_path if results else '',
                'timestamp': datetime.now().isoformat(),
                'average_inference_time_ms': sum(r.inference_time_ms for r in results) / len(results) if results else 0
            },
            'results': [r.to_dict() for r in results]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_jsonl(self, results: List[InferenceResult], output_file: Path) -> None:
        """Save results as JSONL."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result.to_dict(), f, ensure_ascii=False)
                f.write('\n')
    
    def _save_csv(self, results: List[InferenceResult], output_file: Path) -> None:
        """Save results as CSV."""
        if not results:
            return
        
        fieldnames = [
            'text', 'predicted_label', 'confidence', 'inference_time_ms',
            'model_name', 'top_1_label', 'top_1_confidence',
            'top_2_label', 'top_2_confidence', 'top_3_label', 'top_3_confidence'
        ]
        
        # Add true_label if available
        if hasattr(results[0], 'true_label'):
            fieldnames.insert(2, 'true_label')
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    'text': result.text,
                    'predicted_label': result.predicted_label,
                    'confidence': result.confidence,
                    'inference_time_ms': result.inference_time_ms,
                    'model_name': result.model_name
                }
                
                # Add true label if available
                if hasattr(result, 'true_label'):
                    row['true_label'] = result.true_label
                
                # Add top-k predictions
                for i, (label, conf) in enumerate(result.top_k_predictions[:3], 1):
                    row[f'top_{i}_label'] = label
                    row[f'top_{i}_confidence'] = conf
                
                writer.writerow(row)
    
    def generate_summary_report(
        self,
        results: List[InferenceResult],
        output_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive summary report."""
        if not results:
            return {}
        
        # Calculate statistics
        total_samples = len(results)
        avg_inference_time = sum(r.inference_time_ms for r in results) / total_samples
        avg_confidence = sum(r.confidence for r in results) / total_samples
        
        # Count predictions per class
        class_counts = {}
        for result in results:
            label = result.predicted_label
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Calculate confidence distribution
        confidence_ranges = {
            'very_high (>0.9)': sum(1 for r in results if r.confidence > 0.9),
            'high (0.7-0.9)': sum(1 for r in results if 0.7 <= r.confidence <= 0.9),
            'medium (0.5-0.7)': sum(1 for r in results if 0.5 <= r.confidence < 0.7),
            'low (<0.5)': sum(1 for r in results if r.confidence < 0.5)
        }
        
        # Performance metrics
        throughput = 1000 / avg_inference_time if avg_inference_time > 0 else 0
        
        summary = {
            'metadata': {
                'total_samples': total_samples,
                'model_name': results[0].model_name,
                'model_path': results[0].model_path,
                'timestamp': datetime.now().isoformat()
            },
            'performance': {
                'average_inference_time_ms': avg_inference_time,
                'throughput_samples_per_sec': throughput,
                'total_processing_time_ms': sum(r.inference_time_ms for r in results)
            },
            'predictions': {
                'class_distribution': class_counts,
                'average_confidence': avg_confidence,
                'confidence_distribution': confidence_ranges
            }
        }
        
        # Add accuracy if true labels are available
        if hasattr(results[0], 'true_label'):
            correct_predictions = sum(
                1 for r in results 
                if hasattr(r, 'true_label') and r.predicted_label == r.true_label
            )
            accuracy = correct_predictions / total_samples
            summary['evaluation'] = {
                'accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'incorrect_predictions': total_samples - correct_predictions
            }
        
        # Save report if output file specified
        if output_file:
            output_file = Path(output_file)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Summary report saved to {output_file}")
        
        return summary