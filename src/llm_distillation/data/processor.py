"""
Data processing and dataset management module.

This module provides comprehensive data processing capabilities including
dataset loading, preprocessing, format conversion, and batch preparation.
"""

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator, Tuple
from dataclasses import dataclass
import logging
import pickle
import hashlib

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import AutoTokenizer

from ..config import Config
from ..exceptions import ValidationError, ErrorCodes

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    """Configuration for dataset splitting."""
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    stratify: bool = True
    random_state: int = 42
    
    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    lowercase: bool = True
    remove_special_chars: bool = False
    normalize_whitespace: bool = True
    remove_empty: bool = True
    min_length: int = 5
    max_length: int = 1000
    remove_duplicates: bool = True
    balance_classes: bool = False


class DataProcessor:
    """Handles data loading, preprocessing, and format conversion."""
    
    def __init__(self, config: Config):
        self.config = config
        self._cache: Dict[str, Any] = {}
        
    def load_dataset_from_file(
        self,
        file_path: Union[str, Path],
        text_column: str = "text",
        label_column: str = "label"
    ) -> List[Dict[str, Any]]:
        """Load dataset from various file formats."""
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValidationError(
                message=f"Dataset file not found: {file_path}",
                error_code=ErrorCodes.DATA_GENERATION_FAILED
            )
        
        # Generate cache key
        cache_key = self._generate_cache_key(file_path, text_column, label_column)
        if cache_key in self._cache:
            logger.info(f"Loading dataset from cache: {file_path}")
            return self._cache[cache_key]
        
        logger.info(f"Loading dataset from file: {file_path}")
        
        try:
            if file_path.suffix.lower() == '.jsonl':
                samples = self._load_jsonl(file_path, text_column, label_column)
            elif file_path.suffix.lower() == '.json':
                samples = self._load_json(file_path, text_column, label_column)
            elif file_path.suffix.lower() == '.csv':
                samples = self._load_csv(file_path, text_column, label_column)
            elif file_path.suffix.lower() == '.parquet':
                samples = self._load_parquet(file_path, text_column, label_column)
            else:
                raise ValidationError(
                    message=f"Unsupported file format: {file_path.suffix}",
                    error_code=ErrorCodes.DATA_GENERATION_FAILED
                )
            
            # Cache the result
            self._cache[cache_key] = samples
            
            logger.info(f"Loaded {len(samples)} samples from {file_path}")
            return samples
            
        except Exception as e:
            raise ValidationError(
                message=f"Failed to load dataset from {file_path}: {str(e)}",
                error_code=ErrorCodes.DATA_GENERATION_FAILED,
                original_error=e
            )
    
    def _generate_cache_key(self, file_path: Path, text_column: str, label_column: str) -> str:
        """Generate cache key for dataset."""
        content = f"{file_path}:{text_column}:{label_column}:{file_path.stat().st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_jsonl(self, file_path: Path, text_column: str, label_column: str) -> List[Dict[str, Any]]:
        """Load JSONL file."""
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if text_column in data and label_column in data:
                        samples.append({
                            'text': data[text_column],
                            'label': data[label_column],
                            'source': 'file',
                            'line_number': line_num
                        })
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    continue
        return samples
    
    def _load_json(self, file_path: Path, text_column: str, label_column: str) -> List[Dict[str, Any]]:
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and text_column in item and label_column in item:
                    samples.append({
                        'text': item[text_column],
                        'label': item[label_column],
                        'source': 'file'
                    })
        elif isinstance(data, dict):
            if text_column in data and label_column in data:
                # Single sample
                samples.append({
                    'text': data[text_column],
                    'label': data[label_column],
                    'source': 'file'
                })
        
        return samples
    
    def _load_csv(self, file_path: Path, text_column: str, label_column: str) -> List[Dict[str, Any]]:
        """Load CSV file."""
        df = pd.read_csv(file_path)
        
        if text_column not in df.columns or label_column not in df.columns:
            raise ValidationError(
                message=f"Required columns not found. Expected: {text_column}, {label_column}. "
                       f"Found: {list(df.columns)}",
                error_code=ErrorCodes.DATA_GENERATION_FAILED
            )
        
        samples = []
        for idx, row in df.iterrows():
            samples.append({
                'text': str(row[text_column]),
                'label': str(row[label_column]),
                'source': 'file',
                'row_index': idx
            })
        
        return samples
    
    def _load_parquet(self, file_path: Path, text_column: str, label_column: str) -> List[Dict[str, Any]]:
        """Load Parquet file."""
        df = pd.read_parquet(file_path)
        
        if text_column not in df.columns or label_column not in df.columns:
            raise ValidationError(
                message=f"Required columns not found. Expected: {text_column}, {label_column}. "
                       f"Found: {list(df.columns)}",
                error_code=ErrorCodes.DATA_GENERATION_FAILED
            )
        
        samples = []
        for idx, row in df.iterrows():
            samples.append({
                'text': str(row[text_column]),
                'label': str(row[label_column]),
                'source': 'file',
                'row_index': idx
            })
        
        return samples
    
    def preprocess_samples(
        self,
        samples: List[Dict[str, Any]],
        config: Optional[PreprocessingConfig] = None
    ) -> List[Dict[str, Any]]:
        """Apply preprocessing to samples."""
        
        if config is None:
            config = PreprocessingConfig()
        
        logger.info(f"Preprocessing {len(samples)} samples")
        
        processed_samples = []
        
        for sample in samples:
            text = sample.get('text', '')
            if not text:
                continue
            
            # Apply preprocessing steps
            processed_text = self._preprocess_text(text, config)
            
            # Skip if text becomes too short or empty
            if len(processed_text) < config.min_length:
                continue
            
            # Skip if text is too long
            if len(processed_text) > config.max_length:
                processed_text = processed_text[:config.max_length]
            
            # Create processed sample
            processed_sample = sample.copy()
            processed_sample['text'] = processed_text
            processed_sample['original_text'] = text
            processed_sample['preprocessed'] = True
            
            processed_samples.append(processed_sample)
        
        # Remove duplicates if requested
        if config.remove_duplicates:
            processed_samples = self._remove_duplicates(processed_samples)
        
        # Balance classes if requested
        if config.balance_classes:
            processed_samples = self._balance_classes(processed_samples)
        
        logger.info(f"Preprocessing complete: {len(processed_samples)} samples")
        return processed_samples
    
    def _preprocess_text(self, text: str, config: PreprocessingConfig) -> str:
        """Apply text preprocessing steps."""
        
        # Normalize whitespace
        if config.normalize_whitespace:
            text = ' '.join(text.split())
        
        # Convert to lowercase
        if config.lowercase:
            text = text.lower()
        
        # Remove special characters (optional)
        if config.remove_special_chars:
            import re
            text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    def _remove_duplicates(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate samples based on text content."""
        seen_texts = set()
        unique_samples = []
        
        for sample in samples:
            text = sample.get('text', '')
            if text not in seen_texts:
                seen_texts.add(text)
                unique_samples.append(sample)
        
        removed_count = len(samples) - len(unique_samples)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate samples")
        
        return unique_samples
    
    def _balance_classes(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Balance classes by undersampling majority classes."""
        from collections import Counter
        
        # Count samples per class
        class_counts = Counter(sample.get('label', '') for sample in samples)
        
        if not class_counts:
            return samples
        
        # Find minimum class count
        min_count = min(class_counts.values())
        
        # Sample equal number from each class
        class_samples = {}
        for sample in samples:
            label = sample.get('label', '')
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(sample)
        
        balanced_samples = []
        for label, label_samples in class_samples.items():
            # Randomly sample min_count samples
            if len(label_samples) > min_count:
                import random
                random.shuffle(label_samples)
                label_samples = label_samples[:min_count]
            
            balanced_samples.extend(label_samples)
        
        logger.info(f"Balanced dataset: {len(balanced_samples)} samples ({min_count} per class)")
        return balanced_samples
    
    def split_dataset(
        self,
        samples: List[Dict[str, Any]],
        split_config: Optional[DatasetSplit] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset into train, validation, and test sets."""
        
        if split_config is None:
            split_config = DatasetSplit()
        
        if len(samples) < 3:
            raise ValidationError(
                message="Dataset too small to split (need at least 3 samples)",
                error_code=ErrorCodes.DATA_GENERATION_FAILED
            )
        
        # Extract labels for stratification
        labels = [sample.get('label', '') for sample in samples]
        
        # First split: separate test set
        train_val_samples, test_samples, train_val_labels, test_labels = train_test_split(
            samples,
            labels,
            test_size=split_config.test_ratio,
            stratify=labels if split_config.stratify else None,
            random_state=split_config.random_state
        )
        
        # Second split: separate train and validation
        val_ratio_adjusted = split_config.val_ratio / (split_config.train_ratio + split_config.val_ratio)
        
        if val_ratio_adjusted > 0:
            train_samples, val_samples = train_test_split(
                train_val_samples,
                test_size=val_ratio_adjusted,
                stratify=train_val_labels if split_config.stratify else None,
                random_state=split_config.random_state
            )
        else:
            train_samples = train_val_samples
            val_samples = []
        
        logger.info(f"Dataset split - Train: {len(train_samples)}, "
                   f"Val: {len(val_samples)}, Test: {len(test_samples)}")
        
        return train_samples, val_samples, test_samples
    
    def save_samples(
        self,
        samples: List[Dict[str, Any]],
        output_path: Union[str, Path],
        format: str = "jsonl"
    ) -> None:
        """Save samples to file in specified format."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "jsonl":
            self._save_jsonl(samples, output_path)
        elif format.lower() == "json":
            self._save_json(samples, output_path)
        elif format.lower() == "csv":
            self._save_csv(samples, output_path)
        elif format.lower() == "parquet":
            self._save_parquet(samples, output_path)
        else:
            raise ValidationError(
                message=f"Unsupported save format: {format}",
                error_code=ErrorCodes.DATA_GENERATION_FAILED
            )
        
        logger.info(f"Saved {len(samples)} samples to {output_path}")
    
    def _save_jsonl(self, samples: List[Dict[str, Any]], output_path: Path) -> None:
        """Save samples as JSONL."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    def _save_json(self, samples: List[Dict[str, Any]], output_path: Path) -> None:
        """Save samples as JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
    
    def _save_csv(self, samples: List[Dict[str, Any]], output_path: Path) -> None:
        """Save samples as CSV."""
        if not samples:
            return
        
        # Get all unique keys
        all_keys = set()
        for sample in samples:
            all_keys.update(sample.keys())
        
        all_keys = sorted(all_keys)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(samples)
    
    def _save_parquet(self, samples: List[Dict[str, Any]], output_path: Path) -> None:
        """Save samples as Parquet."""
        df = pd.DataFrame(samples)
        df.to_parquet(output_path, index=False)
    
    def clear_cache(self) -> None:
        """Clear the dataset cache."""
        self._cache.clear()
        logger.info("Dataset cache cleared")


class DatasetProcessor:
    """Advanced dataset processing for model training."""
    
    def __init__(self, config: Config, tokenizer: Optional[AutoTokenizer] = None):
        self.config = config
        self.tokenizer = tokenizer
        self.label_encoder = LabelEncoder()
        
    def prepare_for_training(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: Optional[int] = None
    ) -> DatasetDict:
        """Prepare samples for model training."""
        
        if tokenizer is None:
            tokenizer = self.tokenizer
        
        if tokenizer is None:
            raise ValidationError(
                message="Tokenizer is required for training preparation",
                error_code=ErrorCodes.DATA_GENERATION_FAILED
            )
        
        if max_length is None:
            max_length = self.config.max_sequence_length
        
        # Extract texts and labels
        texts = [sample.get('text', '') for sample in samples]
        labels = [sample.get('label', '') for sample in samples]
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Create mapping
        label_to_id = {label: idx for idx, label in enumerate(self.label_encoder.classes_)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        
        # Tokenize texts
        logger.info(f"Tokenizing {len(texts)} samples with max_length={max_length}")
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,  # Let the data collator handle padding dynamically
            max_length=max_length,
            return_tensors=None  # Return lists, not tensors for dynamic batching
        )
        
        # Create dataset
        dataset_dict = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': encoded_labels,  # Keep as list for consistency
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Store metadata
        dataset.label_to_id = label_to_id
        dataset.id_to_label = id_to_label
        dataset.num_labels = len(self.label_encoder.classes_)
        
        return DatasetDict({'train': dataset})
    
    def create_training_splits(
        self,
        dataset: Dataset,
        split_config: Optional[DatasetSplit] = None
    ) -> DatasetDict:
        """Create train/val/test splits from a single dataset."""
        
        if split_config is None:
            split_config = DatasetSplit()
        
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(total_size * split_config.train_ratio)
        val_size = int(total_size * split_config.val_ratio)
        test_size = total_size - train_size - val_size
        
        # Perform splits
        splits = {}
        
        if split_config.stratify and 'labels' in dataset.features:
            # Stratified split
            labels = dataset['labels']
            indices = list(range(len(dataset)))
            
            # First split: train vs (val + test)
            train_indices, temp_indices = train_test_split(
                indices,
                test_size=(val_size + test_size),
                stratify=[labels[i] for i in indices],
                random_state=split_config.random_state
            )
            
            # Second split: val vs test
            if val_size > 0 and test_size > 0:
                temp_labels = [labels[i] for i in temp_indices]
                val_indices, test_indices = train_test_split(
                    temp_indices,
                    test_size=test_size / (val_size + test_size),
                    stratify=temp_labels,
                    random_state=split_config.random_state
                )
            elif val_size > 0:
                val_indices = temp_indices
                test_indices = []
            else:
                val_indices = []
                test_indices = temp_indices
            
        else:
            # Random split
            indices = list(range(len(dataset)))
            np.random.seed(split_config.random_state)
            np.random.shuffle(indices)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
        
        # Create split datasets
        splits['train'] = dataset.select(train_indices)
        
        if val_indices:
            splits['validation'] = dataset.select(val_indices)
        
        if test_indices:
            splits['test'] = dataset.select(test_indices)
        
        dataset_dict = DatasetDict(splits)
        
        # Preserve metadata
        for split_name, split_dataset in dataset_dict.items():
            if hasattr(dataset, 'label_to_id'):
                split_dataset.label_to_id = dataset.label_to_id
            if hasattr(dataset, 'id_to_label'):
                split_dataset.id_to_label = dataset.id_to_label
            if hasattr(dataset, 'num_labels'):
                split_dataset.num_labels = dataset.num_labels
        
        logger.info(f"Created splits: {[f'{k}={len(v)}' for k, v in dataset_dict.items()]}")
        
        return dataset_dict
    
    def create_data_collator(self, tokenizer: AutoTokenizer):
        """Create data collator for dynamic padding."""
        from transformers import DataCollatorWithPadding
        
        return DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            max_length=self.config.max_sequence_length,
            pad_to_multiple_of=8,  # For efficiency on TPUs
            return_tensors="pt"
        )
    
    def get_class_weights(self, labels: List[str]) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets."""
        from collections import Counter
        from sklearn.utils.class_weight import compute_class_weight
        
        # Count label frequencies
        label_counts = Counter(labels)
        unique_labels = list(label_counts.keys())
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.array(unique_labels),
            y=labels
        )
        
        return torch.tensor(class_weights, dtype=torch.float32)
    
    def analyze_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """Analyze dataset properties and statistics."""
        
        analysis = {
            'total_samples': len(dataset),
            'features': list(dataset.features.keys()),
        }
        
        # Text length analysis
        if 'texts' in dataset.features:
            text_lengths = [len(text) for text in dataset['texts']]
            analysis['text_length'] = {
                'mean': np.mean(text_lengths),
                'std': np.std(text_lengths),
                'min': np.min(text_lengths),
                'max': np.max(text_lengths),
                'median': np.median(text_lengths)
            }
        
        # Label distribution
        if 'labels' in dataset.features or 'original_labels' in dataset.features:
            labels = dataset.get('original_labels', dataset.get('labels', []))
            from collections import Counter
            label_counts = Counter(labels)
            
            analysis['label_distribution'] = dict(label_counts)
            analysis['num_classes'] = len(label_counts)
            analysis['class_balance'] = min(label_counts.values()) / max(label_counts.values()) if label_counts else 0
        
        # Token length analysis (if tokenized)
        if 'input_ids' in dataset.features:
            token_lengths = [len(ids) for ids in dataset['input_ids']]
            analysis['token_length'] = {
                'mean': np.mean(token_lengths),
                'std': np.std(token_lengths),
                'min': np.min(token_lengths),
                'max': np.max(token_lengths),
                'median': np.median(token_lengths)
            }
        
        return analysis