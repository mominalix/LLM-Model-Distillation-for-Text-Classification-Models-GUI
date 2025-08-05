"""
Data validation and quality assessment module.

This module provides comprehensive validation of generated text data,
including quality scoring, bias detection, and diversity metrics.
"""

import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import statistics

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams

from ..config import Config
from ..exceptions import ValidationError, ErrorCodes

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    logger.warning("Failed to download NLTK data")


@dataclass
class QualityMetrics:
    """Quality metrics for a dataset or sample."""
    
    # Basic metrics
    total_samples: int = 0
    average_length: float = 0.0
    length_std: float = 0.0
    vocabulary_size: int = 0
    
    # Diversity metrics
    lexical_diversity: float = 0.0  # TTR - Type-Token Ratio
    semantic_diversity: float = 0.0
    structural_diversity: float = 0.0
    
    # Quality scores
    overall_quality: float = 0.0
    readability_score: float = 0.0
    coherence_score: float = 0.0
    
    # Class distribution
    class_balance: float = 0.0  # How balanced are the classes
    class_counts: Dict[str, int] = field(default_factory=dict)
    
    # Bias and safety metrics
    bias_score: float = 0.0
    toxicity_score: float = 0.0
    pii_detected: bool = False
    
    # Uniqueness metrics
    duplicate_ratio: float = 0.0
    near_duplicate_ratio: float = 0.0
    
    # N-gram diversity
    distinct_1: float = 0.0  # Distinct unigrams
    distinct_2: float = 0.0  # Distinct bigrams
    distinct_3: float = 0.0  # Distinct trigrams


class DataValidator:
    """Comprehensive data validation and quality assessment."""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize PII patterns
        self._pii_patterns = self._compile_pii_patterns()
        
        # Initialize bias detection patterns
        self._bias_patterns = self._compile_bias_patterns()
        
        # Initialize toxicity keywords (simple implementation)
        self._toxicity_keywords = self._load_toxicity_keywords()
        
        # Cache for repeated calculations
        self._tfidf_cache: Dict[str, Any] = {}
        
    def _compile_pii_patterns(self) -> List[Tuple[str, re.Pattern]]:
        """Compile regex patterns for PII detection."""
        patterns = [
            # Email addresses
            ("email", re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')),
            
            # Phone numbers (various formats)
            ("phone", re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')),
            
            # Social Security Numbers
            ("ssn", re.compile(r'\b\d{3}-\d{2}-\d{4}\b')),
            
            # Credit card numbers (simple pattern)
            ("credit_card", re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')),
            
            # URLs
            ("url", re.compile(r'https?://[^\s]+|www\.[^\s]+')),
            
            # IP addresses
            ("ip", re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')),
            
            # Names (simple pattern - very basic)
            ("name", re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b')),
        ]
        
        return patterns
    
    def _compile_bias_patterns(self) -> List[Tuple[str, re.Pattern]]:
        """Compile patterns for bias detection."""
        # Simplified bias detection - in production, use more sophisticated methods
        bias_terms = [
            # Gender bias
            r'\b(he|she)\s+(is|was)\s+(better|worse|smarter|dumber)',
            r'\b(men|women)\s+(are|tend to be)\s+',
            
            # Age bias
            r'\b(old|young)\s+people\s+(are|can\'t|cannot)',
            r'\b(elderly|seniors)\s+(should|shouldn\'t)',
            
            # Racial/ethnic bias (very careful with this)
            r'\b(people of color|minorities)\s+(are|tend to)',
            
            # Religious bias
            r'\b(christians|muslims|jews|buddhists|hindus)\s+(are|believe)',
        ]
        
        patterns = []
        for i, pattern in enumerate(bias_terms):
            try:
                patterns.append((f"bias_{i}", re.compile(pattern, re.IGNORECASE)))
            except re.error as e:
                logger.warning(f"Invalid bias pattern {pattern}: {e}")
        
        return patterns
    
    def _load_toxicity_keywords(self) -> Set[str]:
        """Load simple toxicity keyword list."""
        # In production, use a comprehensive toxicity detection model
        # This is a minimal placeholder
        toxic_words = {
            'hate', 'stupid', 'idiot', 'moron', 'dumb', 'kill', 'die', 'murder'
        }
        return toxic_words
    
    def assess_sample_quality(self, sample: Dict[str, Any]) -> float:
        """Assess the quality of a single sample."""
        
        text = sample.get('text', '')
        if not text:
            return 0.0
        
        scores = []
        
        # Length appropriateness (0.0 - 1.0)
        length_score = self._assess_length_quality(text)
        scores.append(length_score)
        
        # Readability (0.0 - 1.0)
        readability_score = self._assess_readability(text)
        scores.append(readability_score)
        
        # Coherence (0.0 - 1.0)
        coherence_score = self._assess_coherence(text)
        scores.append(coherence_score)
        
        # Safety checks (binary - 0.0 or 1.0)
        safety_score = self._assess_safety(text)
        scores.append(safety_score)
        
        # Content richness (0.0 - 1.0)
        richness_score = self._assess_content_richness(text)
        scores.append(richness_score)
        
        # Calculate weighted average
        weights = [0.2, 0.2, 0.2, 0.3, 0.1]  # Safety gets highest weight
        quality_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return min(max(quality_score, 0.0), 1.0)
    
    def _assess_length_quality(self, text: str) -> float:
        """Assess if text length is appropriate."""
        length = len(text)
        
        if length < self.config.min_text_length:
            return 0.0
        elif length > self.config.max_text_length:
            return 0.0
        
        # Optimal range scoring
        min_good = self.config.min_text_length * 2
        max_good = self.config.max_text_length * 0.8
        
        if min_good <= length <= max_good:
            return 1.0
        elif length < min_good:
            return length / min_good
        else:
            return max_good / length
    
    def _assess_readability(self, text: str) -> float:
        """Assess text readability using simple metrics."""
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            if not sentences or not words:
                return 0.0
            
            # Average sentence length
            avg_sentence_length = len(words) / len(sentences)
            
            # Average word length
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Simple readability score (normalized)
            # Prefer moderate sentence and word lengths
            sentence_score = 1.0 / (1.0 + abs(avg_sentence_length - 15) / 10)
            word_score = 1.0 / (1.0 + abs(avg_word_length - 5) / 3)
            
            return (sentence_score + word_score) / 2
            
        except Exception as e:
            logger.warning(f"Error calculating readability: {e}")
            return 0.5
    
    def _assess_coherence(self, text: str) -> float:
        """Assess text coherence and structure."""
        try:
            sentences = sent_tokenize(text)
            
            if len(sentences) < 2:
                return 0.8  # Single sentences can still be coherent
            
            # Check for basic coherence indicators
            coherence_indicators = 0
            
            # Consistent tense
            if self._has_consistent_tense(text):
                coherence_indicators += 1
            
            # Proper punctuation
            if self._has_proper_punctuation(text):
                coherence_indicators += 1
            
            # Logical flow (simple check)
            if self._has_logical_flow(sentences):
                coherence_indicators += 1
            
            # Vocabulary consistency
            if self._has_vocabulary_consistency(text):
                coherence_indicators += 1
            
            return coherence_indicators / 4
            
        except Exception as e:
            logger.warning(f"Error assessing coherence: {e}")
            return 0.5
    
    def _assess_safety(self, text: str) -> float:
        """Assess text safety (PII, toxicity, bias)."""
        
        # PII detection
        if self._contains_pii(text):
            return 0.0
        
        # Toxicity detection
        if self._contains_toxicity(text):
            return 0.0
        
        # Bias detection
        bias_score = self._assess_bias(text)
        if bias_score > 0.5:  # High bias
            return 0.0
        
        return 1.0
    
    def _assess_content_richness(self, text: str) -> float:
        """Assess content richness and informativeness."""
        try:
            words = word_tokenize(text.lower())
            
            if not words:
                return 0.0
            
            # Remove stopwords and punctuation
            try:
                stop_words = set(stopwords.words('english'))
            except:
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            content_words = [
                word for word in words 
                if word not in stop_words and word not in string.punctuation and len(word) > 2
            ]
            
            if not content_words:
                return 0.0
            
            # Vocabulary diversity
            unique_words = set(content_words)
            diversity = len(unique_words) / len(content_words)
            
            # Information density (content words vs total words)
            density = len(content_words) / len(words)
            
            return (diversity + density) / 2
            
        except Exception as e:
            logger.warning(f"Error assessing content richness: {e}")
            return 0.5
    
    def _contains_pii(self, text: str) -> bool:
        """Check if text contains PII."""
        for pii_type, pattern in self._pii_patterns:
            if pattern.search(text):
                logger.warning(f"PII detected: {pii_type}")
                return True
        return False
    
    def _contains_toxicity(self, text: str) -> bool:
        """Check if text contains toxic content."""
        words = word_tokenize(text.lower())
        for word in words:
            if word in self._toxicity_keywords:
                logger.warning(f"Toxic word detected: {word}")
                return True
        return False
    
    def _assess_bias(self, text: str) -> float:
        """Assess potential bias in text."""
        bias_count = 0
        for bias_type, pattern in self._bias_patterns:
            if pattern.search(text):
                bias_count += 1
        
        # Normalize bias score
        max_possible_bias = len(self._bias_patterns)
        return bias_count / max_possible_bias if max_possible_bias > 0 else 0.0
    
    def _has_consistent_tense(self, text: str) -> bool:
        """Check for consistent verb tense (simplified)."""
        # Very simple implementation - in production use POS tagging
        past_indicators = ['was', 'were', 'had', 'did', 'ed']
        present_indicators = ['is', 'are', 'have', 'do', 'does']
        future_indicators = ['will', 'shall', 'going to']
        
        text_lower = text.lower()
        
        past_count = sum(1 for indicator in past_indicators if indicator in text_lower)
        present_count = sum(1 for indicator in present_indicators if indicator in text_lower)
        future_count = sum(1 for indicator in future_indicators if indicator in text_lower)
        
        total_indicators = past_count + present_count + future_count
        if total_indicators == 0:
            return True
        
        # Check if one tense dominates
        max_count = max(past_count, present_count, future_count)
        return (max_count / total_indicators) > 0.7
    
    def _has_proper_punctuation(self, text: str) -> bool:
        """Check for proper punctuation."""
        # Basic checks
        if not text.strip():
            return False
        
        # Should end with proper punctuation
        if not text.strip()[-1] in '.!?':
            return False
        
        # Check for balanced quotes and parentheses
        quote_count = text.count('"')
        paren_open = text.count('(')
        paren_close = text.count(')')
        
        return quote_count % 2 == 0 and paren_open == paren_close
    
    def _has_logical_flow(self, sentences: List[str]) -> bool:
        """Check for logical flow between sentences (simplified)."""
        if len(sentences) < 2:
            return True
        
        # Simple coherence check using sentence similarity
        try:
            # Use simple word overlap as similarity metric
            similarities = []
            for i in range(len(sentences) - 1):
                words1 = set(word_tokenize(sentences[i].lower()))
                words2 = set(word_tokenize(sentences[i + 1].lower()))
                
                if not words1 or not words2:
                    continue
                
                overlap = len(words1.intersection(words2))
                total = len(words1.union(words2))
                similarity = overlap / total if total > 0 else 0
                similarities.append(similarity)
            
            if not similarities:
                return True
            
            # Good flow should have some but not too much similarity
            avg_similarity = statistics.mean(similarities)
            return 0.1 <= avg_similarity <= 0.7
            
        except Exception as e:
            logger.warning(f"Error checking logical flow: {e}")
            return True
    
    def _has_vocabulary_consistency(self, text: str) -> bool:
        """Check for vocabulary consistency."""
        try:
            words = word_tokenize(text.lower())
            if len(words) < 5:
                return True
            
            # Check for reasonable vocabulary diversity
            unique_words = set(words)
            diversity = len(unique_words) / len(words)
            
            # Should be diverse but not too diverse (indicating randomness)
            return 0.3 <= diversity <= 0.9
            
        except Exception as e:
            logger.warning(f"Error checking vocabulary consistency: {e}")
            return True
    
    def calculate_dataset_metrics(self, samples: List[Dict[str, Any]]) -> QualityMetrics:
        """Calculate comprehensive metrics for a dataset."""
        
        if not samples:
            return QualityMetrics()
        
        texts = [sample.get('text', '') for sample in samples]
        labels = [sample.get('label', '') for sample in samples]
        
        # Basic metrics
        lengths = [len(text) for text in texts if text]
        total_samples = len(samples)
        average_length = statistics.mean(lengths) if lengths else 0
        length_std = statistics.stdev(lengths) if len(lengths) > 1 else 0
        
        # Vocabulary metrics
        all_words = []
        for text in texts:
            if text:
                words = word_tokenize(text.lower())
                all_words.extend(words)
        
        vocabulary_size = len(set(all_words))
        
        # Diversity metrics
        lexical_diversity = self._calculate_lexical_diversity(all_words)
        semantic_diversity = self._calculate_semantic_diversity(texts)
        structural_diversity = self._calculate_structural_diversity(texts)
        
        # Quality scores
        quality_scores = [self.assess_sample_quality(sample) for sample in samples]
        overall_quality = statistics.mean(quality_scores) if quality_scores else 0
        
        readability_scores = [self._assess_readability(text) for text in texts if text]
        readability_score = statistics.mean(readability_scores) if readability_scores else 0
        
        coherence_scores = [self._assess_coherence(text) for text in texts if text]
        coherence_score = statistics.mean(coherence_scores) if coherence_scores else 0
        
        # Class distribution
        class_counts = Counter(labels)
        class_balance = self._calculate_class_balance(class_counts)
        
        # Safety metrics
        bias_scores = [self._assess_bias(text) for text in texts if text]
        bias_score = statistics.mean(bias_scores) if bias_scores else 0
        
        pii_detected = any(self._contains_pii(text) for text in texts if text)
        
        toxicity_scores = [1.0 if self._contains_toxicity(text) else 0.0 for text in texts if text]
        toxicity_score = statistics.mean(toxicity_scores) if toxicity_scores else 0
        
        # Uniqueness metrics
        duplicate_ratio = self._calculate_duplicate_ratio(texts)
        near_duplicate_ratio = self._calculate_near_duplicate_ratio(texts)
        
        # N-gram diversity
        distinct_1, distinct_2, distinct_3 = self._calculate_distinct_ngrams(all_words)
        
        return QualityMetrics(
            total_samples=total_samples,
            average_length=average_length,
            length_std=length_std,
            vocabulary_size=vocabulary_size,
            lexical_diversity=lexical_diversity,
            semantic_diversity=semantic_diversity,
            structural_diversity=structural_diversity,
            overall_quality=overall_quality,
            readability_score=readability_score,
            coherence_score=coherence_score,
            class_balance=class_balance,
            class_counts=dict(class_counts),
            bias_score=bias_score,
            toxicity_score=toxicity_score,
            pii_detected=pii_detected,
            duplicate_ratio=duplicate_ratio,
            near_duplicate_ratio=near_duplicate_ratio,
            distinct_1=distinct_1,
            distinct_2=distinct_2,
            distinct_3=distinct_3
        )
    
    def _calculate_lexical_diversity(self, words: List[str]) -> float:
        """Calculate Type-Token Ratio (TTR)."""
        if not words:
            return 0.0
        
        unique_words = set(words)
        return len(unique_words) / len(words)
    
    def _calculate_semantic_diversity(self, texts: List[str]) -> float:
        """Calculate semantic diversity using TF-IDF similarity."""
        if len(texts) < 2:
            return 1.0
        
        try:
            # Filter out empty texts
            valid_texts = [text for text in texts if text.strip()]
            if len(valid_texts) < 2:
                return 1.0
            
            # Use TF-IDF to measure semantic similarity
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(valid_texts)
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(tfidf_matrix)
            
            # Extract upper triangle (avoid diagonal and duplicates)
            n = similarities.shape[0]
            similarity_values = []
            for i in range(n):
                for j in range(i + 1, n):
                    similarity_values.append(similarities[i, j])
            
            if not similarity_values:
                return 1.0
            
            # Diversity is inverse of average similarity
            avg_similarity = statistics.mean(similarity_values)
            return 1.0 - avg_similarity
            
        except Exception as e:
            logger.warning(f"Error calculating semantic diversity: {e}")
            return 0.5
    
    def _calculate_structural_diversity(self, texts: List[str]) -> float:
        """Calculate structural diversity (sentence count, punctuation patterns)."""
        if not texts:
            return 0.0
        
        try:
            sentence_counts = []
            punct_patterns = []
            
            for text in texts:
                if not text:
                    continue
                
                # Sentence count
                sentences = sent_tokenize(text)
                sentence_counts.append(len(sentences))
                
                # Punctuation pattern
                punct_chars = [c for c in text if c in string.punctuation]
                punct_patterns.append(''.join(punct_chars))
            
            # Diversity in sentence counts
            sentence_diversity = 0.0
            if sentence_counts:
                unique_counts = len(set(sentence_counts))
                sentence_diversity = unique_counts / len(sentence_counts)
            
            # Diversity in punctuation patterns
            punct_diversity = 0.0
            if punct_patterns:
                unique_patterns = len(set(punct_patterns))
                punct_diversity = unique_patterns / len(punct_patterns)
            
            return (sentence_diversity + punct_diversity) / 2
            
        except Exception as e:
            logger.warning(f"Error calculating structural diversity: {e}")
            return 0.5
    
    def _calculate_class_balance(self, class_counts: Counter) -> float:
        """Calculate class balance score (1.0 = perfectly balanced)."""
        if not class_counts:
            return 0.0
        
        counts = list(class_counts.values())
        if len(counts) == 1:
            return 1.0
        
        min_count = min(counts)
        max_count = max(counts)
        
        # Perfect balance when min_count == max_count
        return min_count / max_count if max_count > 0 else 0.0
    
    def _calculate_duplicate_ratio(self, texts: List[str]) -> float:
        """Calculate ratio of exact duplicates."""
        if not texts:
            return 0.0
        
        text_counts = Counter(texts)
        duplicates = sum(count - 1 for count in text_counts.values() if count > 1)
        
        return duplicates / len(texts)
    
    def _calculate_near_duplicate_ratio(self, texts: List[str], threshold: float = 0.9) -> float:
        """Calculate ratio of near-duplicates using simple similarity."""
        if len(texts) < 2:
            return 0.0
        
        try:
            near_duplicates = 0
            total_pairs = 0
            
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    text1, text2 = texts[i], texts[j]
                    if not text1 or not text2:
                        continue
                    
                    # Simple word-based similarity
                    words1 = set(word_tokenize(text1.lower()))
                    words2 = set(word_tokenize(text2.lower()))
                    
                    if not words1 or not words2:
                        continue
                    
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity >= threshold:
                        near_duplicates += 1
                    
                    total_pairs += 1
            
            return near_duplicates / total_pairs if total_pairs > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating near-duplicate ratio: {e}")
            return 0.0
    
    def _calculate_distinct_ngrams(self, words: List[str]) -> Tuple[float, float, float]:
        """Calculate distinct n-gram ratios."""
        if not words:
            return 0.0, 0.0, 0.0
        
        try:
            # Distinct unigrams
            unigrams = words
            distinct_1 = len(set(unigrams)) / len(unigrams) if unigrams else 0.0
            
            # Distinct bigrams
            bigrams = list(ngrams(words, 2))
            distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
            
            # Distinct trigrams
            trigrams = list(ngrams(words, 3))
            distinct_3 = len(set(trigrams)) / len(trigrams) if trigrams else 0.0
            
            return distinct_1, distinct_2, distinct_3
            
        except Exception as e:
            logger.warning(f"Error calculating distinct n-grams: {e}")
            return 0.0, 0.0, 0.0
    
    def validate_dataset_balance(
        self,
        samples: List[Dict[str, Any]],
        target_classes: List[str],
        tolerance: float = 0.1
    ) -> bool:
        """Validate that dataset is balanced within tolerance."""
        
        class_counts = Counter(sample.get('label') for sample in samples)
        
        # Check all target classes are present
        for class_label in target_classes:
            if class_label not in class_counts:
                return False
        
        # Check balance
        counts = [class_counts[label] for label in target_classes]
        if not counts:
            return False
        
        min_count = min(counts)
        max_count = max(counts)
        
        # Calculate imbalance ratio
        imbalance = (max_count - min_count) / max_count if max_count > 0 else 1.0
        
        return imbalance <= tolerance
    
    def filter_low_quality_samples(
        self,
        samples: List[Dict[str, Any]],
        quality_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Filter out samples below quality threshold."""
        
        filtered_samples = []
        
        for sample in samples:
            quality_score = self.assess_sample_quality(sample)
            if quality_score >= quality_threshold:
                sample['quality_score'] = quality_score
                filtered_samples.append(sample)
        
        logger.info(f"Filtered {len(filtered_samples)}/{len(samples)} samples above quality threshold {quality_threshold}")
        
        return filtered_samples