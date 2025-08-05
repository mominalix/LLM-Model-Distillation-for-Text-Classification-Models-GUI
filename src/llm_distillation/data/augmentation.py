"""
Data augmentation module for improving dataset diversity and balance.

This module provides various augmentation strategies to enhance synthetic
datasets including paraphrasing, back-translation, and structural variations.
"""

import random
import re
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass
import logging

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag

from ..config import Config
from ..exceptions import DataGenerationError, ErrorCodes

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    logger.warning("Failed to download NLTK data for augmentation")


class AugmentationStrategy(str, Enum):
    """Available augmentation strategies."""
    SYNONYM_REPLACEMENT = "synonym_replacement"
    PARAPHRASING = "paraphrasing"
    SENTENCE_REORDERING = "sentence_reordering"
    WORD_INSERTION = "word_insertion"
    WORD_DELETION = "word_deletion"
    PUNCTUATION_VARIATION = "punctuation_variation"
    CASE_VARIATION = "case_variation"
    CONTRACTION_EXPANSION = "contraction_expansion"


@dataclass
class AugmentationConfig:
    """Configuration for augmentation process."""
    strategies: List[AugmentationStrategy]
    augmentation_factor: float = 1.5  # How many augmented samples per original
    preserve_length: bool = False
    preserve_semantics: bool = True
    min_quality_threshold: float = 0.6
    max_changes_per_text: int = 3


class DataAugmenter:
    """Comprehensive data augmentation system."""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Contraction mappings
        self.contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        self.expansions = {v: k for k, v in self.contractions.items()}
        
        # Common word insertions (context-dependent)
        self.insertion_words = {
            'adverbs': ['really', 'very', 'quite', 'extremely', 'particularly', 'especially'],
            'connectors': ['however', 'moreover', 'furthermore', 'additionally', 'also'],
            'emphasis': ['indeed', 'certainly', 'absolutely', 'definitely', 'truly'],
        }
        
        # Punctuation variations
        self.punctuation_variations = {
            '.': ['!', '.', '...'],
            '!': ['!', '!!', '.'],
            '?': ['?', '??', '.'],
        }
    
    def augment_samples(
        self,
        samples: List[Dict[str, Any]],
        config: Optional[AugmentationConfig] = None
    ) -> List[Dict[str, Any]]:
        """Augment a list of samples using various strategies."""
        
        if config is None:
            config = AugmentationConfig(
                strategies=[
                    AugmentationStrategy.SYNONYM_REPLACEMENT,
                    AugmentationStrategy.SENTENCE_REORDERING,
                    AugmentationStrategy.PUNCTUATION_VARIATION
                ]
            )
        
        augmented_samples = []
        
        for sample in samples:
            try:
                # Calculate how many augmented versions to create
                num_augmentations = max(1, int(config.augmentation_factor))
                
                for _ in range(num_augmentations):
                    augmented_sample = self._augment_single_sample(sample, config)
                    if augmented_sample:
                        augmented_samples.append(augmented_sample)
                        
            except Exception as e:
                logger.warning(f"Failed to augment sample: {e}")
                continue
        
        logger.info(f"Generated {len(augmented_samples)} augmented samples from {len(samples)} originals")
        return augmented_samples
    
    def _augment_single_sample(
        self,
        sample: Dict[str, Any],
        config: AugmentationConfig
    ) -> Optional[Dict[str, Any]]:
        """Augment a single sample using random strategies."""
        
        original_text = sample.get('text', '')
        if not original_text:
            return None
        
        # Select random strategies to apply
        num_strategies = min(
            len(config.strategies),
            random.randint(1, config.max_changes_per_text)
        )
        selected_strategies = random.sample(config.strategies, num_strategies)
        
        augmented_text = original_text
        changes_applied = []
        
        for strategy in selected_strategies:
            try:
                if strategy == AugmentationStrategy.SYNONYM_REPLACEMENT:
                    augmented_text, changed = self._apply_synonym_replacement(augmented_text)
                elif strategy == AugmentationStrategy.PARAPHRASING:
                    augmented_text, changed = self._apply_paraphrasing(augmented_text)
                elif strategy == AugmentationStrategy.SENTENCE_REORDERING:
                    augmented_text, changed = self._apply_sentence_reordering(augmented_text)
                elif strategy == AugmentationStrategy.WORD_INSERTION:
                    augmented_text, changed = self._apply_word_insertion(augmented_text)
                elif strategy == AugmentationStrategy.WORD_DELETION:
                    augmented_text, changed = self._apply_word_deletion(augmented_text)
                elif strategy == AugmentationStrategy.PUNCTUATION_VARIATION:
                    augmented_text, changed = self._apply_punctuation_variation(augmented_text)
                elif strategy == AugmentationStrategy.CASE_VARIATION:
                    augmented_text, changed = self._apply_case_variation(augmented_text)
                elif strategy == AugmentationStrategy.CONTRACTION_EXPANSION:
                    augmented_text, changed = self._apply_contraction_expansion(augmented_text)
                
                if changed:
                    changes_applied.append(strategy.value)
                    
            except Exception as e:
                logger.warning(f"Error applying {strategy}: {e}")
                continue
        
        # Check if any changes were made
        if augmented_text == original_text:
            return None
        
        # Validate length constraints
        if config.preserve_length:
            original_length = len(original_text)
            augmented_length = len(augmented_text)
            length_ratio = augmented_length / original_length
            
            if not (0.8 <= length_ratio <= 1.2):
                return None
        
        # Create augmented sample
        augmented_sample = sample.copy()
        augmented_sample.update({
            'text': augmented_text,
            'source': 'augmented',
            'original_text': original_text,
            'augmentation_strategies': changes_applied,
            'confidence': sample.get('confidence', 0.8) * 0.9  # Slightly lower confidence
        })
        
        return augmented_sample
    
    def _apply_synonym_replacement(self, text: str) -> Tuple[str, bool]:
        """Replace words with synonyms using WordNet."""
        try:
            words = word_tokenize(text)
            pos_tags = pos_tag(words)
            
            changed = False
            new_words = []
            
            for word, pos in pos_tags:
                new_word = word
                
                # Only replace content words (nouns, verbs, adjectives, adverbs)
                if pos.startswith(('NN', 'VB', 'JJ', 'RB')) and len(word) > 3:
                    synonyms = self._get_synonyms(word, pos)
                    if synonyms and random.random() < 0.3:  # 30% chance of replacement
                        new_word = random.choice(synonyms)
                        if new_word != word:
                            changed = True
                
                new_words.append(new_word)
            
            return ' '.join(new_words), changed
            
        except Exception as e:
            logger.warning(f"Error in synonym replacement: {e}")
            return text, False
    
    def _get_synonyms(self, word: str, pos: str) -> List[str]:
        """Get synonyms for a word based on its POS tag."""
        try:
            # Map POS tags to WordNet POS
            pos_map = {
                'NN': wordnet.NOUN,
                'VB': wordnet.VERB,
                'JJ': wordnet.ADJ,
                'RB': wordnet.ADV
            }
            
            wn_pos = None
            for tag_prefix, wn_tag in pos_map.items():
                if pos.startswith(tag_prefix):
                    wn_pos = wn_tag
                    break
            
            if not wn_pos:
                return []
            
            synonyms = set()
            for synset in wordnet.synsets(word, pos=wn_pos):
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower() and synonym.isalpha():
                        synonyms.add(synonym)
            
            return list(synonyms)[:5]  # Limit to 5 synonyms
            
        except Exception as e:
            logger.warning(f"Error getting synonyms for {word}: {e}")
            return []
    
    def _apply_paraphrasing(self, text: str) -> Tuple[str, bool]:
        """Apply simple paraphrasing transformations."""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) < 2:
                return text, False
            
            paraphrased_sentences = []
            changed = False
            
            for sentence in sentences:
                paraphrased = self._paraphrase_sentence(sentence)
                if paraphrased != sentence:
                    changed = True
                paraphrased_sentences.append(paraphrased)
            
            return ' '.join(paraphrased_sentences), changed
            
        except Exception as e:
            logger.warning(f"Error in paraphrasing: {e}")
            return text, False
    
    def _paraphrase_sentence(self, sentence: str) -> str:
        """Apply paraphrasing transformations to a sentence."""
        # Simple paraphrasing rules
        paraphrasing_rules = [
            # Active to passive voice (simplified)
            (r'\b(\w+)\s+(is|are|was|were)\s+(\w+ing)\b', r'\3 \2 being done by \1'),
            
            # Change sentence structure
            (r'\bBecause\s+(.+?),\s+(.+)', r'\2 because \1'),
            (r'\b(.+?),\s+but\s+(.+)', r'\2, although \1'),
            
            # Synonym-like replacements
            (r'\bvery\s+(\w+)', r'extremely \1'),
            (r'\bgood\b', 'excellent'),
            (r'\bbad\b', 'poor'),
            (r'\bbig\b', 'large'),
            (r'\bsmall\b', 'tiny'),
        ]
        
        result = sentence
        for pattern, replacement in paraphrasing_rules:
            if random.random() < 0.2:  # 20% chance for each rule
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def _apply_sentence_reordering(self, text: str) -> Tuple[str, bool]:
        """Reorder sentences if there are multiple sentences."""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) < 2:
                return text, False
            
            # Only reorder if we have 2-4 sentences to avoid breaking coherence
            if 2 <= len(sentences) <= 4:
                original_order = list(range(len(sentences)))
                new_order = original_order.copy()
                random.shuffle(new_order)
                
                if new_order != original_order:
                    reordered_sentences = [sentences[i] for i in new_order]
                    return ' '.join(reordered_sentences), True
            
            return text, False
            
        except Exception as e:
            logger.warning(f"Error in sentence reordering: {e}")
            return text, False
    
    def _apply_word_insertion(self, text: str) -> Tuple[str, bool]:
        """Insert additional words for emphasis or clarity."""
        try:
            words = word_tokenize(text)
            if len(words) < 5:
                return text, False
            
            # Find good insertion points (before adjectives, nouns)
            pos_tags = pos_tag(words)
            insertion_points = []
            
            for i, (word, pos) in enumerate(pos_tags):
                if pos.startswith(('JJ', 'NN')) and i > 0:
                    insertion_points.append(i)
            
            if not insertion_points:
                return text, False
            
            # Insert a word at a random point
            insert_pos = random.choice(insertion_points)
            word_type = random.choice(list(self.insertion_words.keys()))
            insert_word = random.choice(self.insertion_words[word_type])
            
            new_words = words[:insert_pos] + [insert_word] + words[insert_pos:]
            return ' '.join(new_words), True
            
        except Exception as e:
            logger.warning(f"Error in word insertion: {e}")
            return text, False
    
    def _apply_word_deletion(self, text: str) -> Tuple[str, bool]:
        """Delete non-essential words."""
        try:
            words = word_tokenize(text)
            if len(words) < 8:  # Don't delete from short texts
                return text, False
            
            # Find deletable words (adverbs, some adjectives)
            pos_tags = pos_tag(words)
            deletable_indices = []
            
            for i, (word, pos) in enumerate(pos_tags):
                # Delete adverbs and some adjectives that aren't essential
                if pos in ['RB', 'RBR', 'RBS'] or (pos.startswith('JJ') and word.lower() in 
                    ['very', 'quite', 'really', 'extremely', 'particularly']):
                    deletable_indices.append(i)
            
            if not deletable_indices:
                return text, False
            
            # Delete one word
            delete_idx = random.choice(deletable_indices)
            new_words = words[:delete_idx] + words[delete_idx + 1:]
            
            return ' '.join(new_words), True
            
        except Exception as e:
            logger.warning(f"Error in word deletion: {e}")
            return text, False
    
    def _apply_punctuation_variation(self, text: str) -> Tuple[str, bool]:
        """Vary punctuation for different emphasis."""
        try:
            changed = False
            result = text
            
            for punct, variations in self.punctuation_variations.items():
                if punct in result and random.random() < 0.3:
                    new_punct = random.choice(variations)
                    if new_punct != punct:
                        result = result.replace(punct, new_punct, 1)  # Replace only first occurrence
                        changed = True
            
            return result, changed
            
        except Exception as e:
            logger.warning(f"Error in punctuation variation: {e}")
            return text, False
    
    def _apply_case_variation(self, text: str) -> Tuple[str, bool]:
        """Apply minor case variations."""
        try:
            # Only apply to texts that are properly capitalized
            if not text or not text[0].isupper():
                return text, False
            
            # Random chance to change first word to lowercase (informal style)
            if random.random() < 0.2:
                words = text.split()
                if words and words[0][0].isupper():
                    words[0] = words[0][0].lower() + words[0][1:]
                    return ' '.join(words), True
            
            return text, False
            
        except Exception as e:
            logger.warning(f"Error in case variation: {e}")
            return text, False
    
    def _apply_contraction_expansion(self, text: str) -> Tuple[str, bool]:
        """Expand or contract words."""
        try:
            changed = False
            result = text
            
            # Randomly choose to expand or contract
            if random.random() < 0.5:
                # Expand contractions
                for contraction, expansion in self.contractions.items():
                    if contraction in result:
                        result = result.replace(contraction, expansion)
                        changed = True
            else:
                # Create contractions
                for expansion, contraction in self.expansions.items():
                    if expansion in result and random.random() < 0.3:
                        result = result.replace(expansion, contraction)
                        changed = True
            
            return result, changed
            
        except Exception as e:
            logger.warning(f"Error in contraction expansion: {e}")
            return text, False
    
    def balance_dataset(
        self,
        samples: List[Dict[str, Any]],
        target_per_class: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Balance dataset by augmenting minority classes."""
        
        # Count samples per class
        class_counts = {}
        class_samples = {}
        
        for sample in samples:
            label = sample.get('label', '')
            if label not in class_counts:
                class_counts[label] = 0
                class_samples[label] = []
            
            class_counts[label] += 1
            class_samples[label].append(sample)
        
        if not class_counts:
            return samples
        
        # Determine target count
        if target_per_class is None:
            target_per_class = max(class_counts.values())
        
        balanced_samples = []
        
        for label, count in class_counts.items():
            label_samples = class_samples[label]
            balanced_samples.extend(label_samples)
            
            # Augment if needed
            if count < target_per_class:
                needed = target_per_class - count
                logger.info(f"Augmenting {needed} samples for class '{label}'")
                
                # Create augmentation config for balancing
                balance_config = AugmentationConfig(
                    strategies=[
                        AugmentationStrategy.SYNONYM_REPLACEMENT,
                        AugmentationStrategy.PUNCTUATION_VARIATION,
                        AugmentationStrategy.WORD_INSERTION
                    ],
                    augmentation_factor=1.0,  # One augmentation per sample
                    preserve_semantics=True
                )
                
                # Repeatedly augment samples until we reach target
                while len([s for s in balanced_samples if s.get('label') == label]) < target_per_class:
                    # Select samples to augment (prefer originals over augmented)
                    candidates = [s for s in label_samples if s.get('source', 'original') == 'original']
                    if not candidates:
                        candidates = label_samples
                    
                    sample_to_augment = random.choice(candidates)
                    augmented = self._augment_single_sample(sample_to_augment, balance_config)
                    
                    if augmented:
                        balanced_samples.append(augmented)
                    else:
                        # Fallback: duplicate with minor modification
                        fallback = sample_to_augment.copy()
                        fallback['source'] = 'duplicated'
                        fallback['confidence'] = fallback.get('confidence', 0.8) * 0.8
                        balanced_samples.append(fallback)
        
        logger.info(f"Balanced dataset: {len(balanced_samples)} total samples")
        return balanced_samples
    
    def create_adversarial_samples(
        self,
        samples: List[Dict[str, Any]],
        num_adversarial: int = 10
    ) -> List[Dict[str, Any]]:
        """Create challenging adversarial samples for robust training."""
        
        adversarial_samples = []
        
        for _ in range(num_adversarial):
            if not samples:
                break
                
            # Select a random sample
            base_sample = random.choice(samples)
            
            # Apply aggressive augmentation
            adversarial_config = AugmentationConfig(
                strategies=[
                    AugmentationStrategy.SYNONYM_REPLACEMENT,
                    AugmentationStrategy.WORD_INSERTION,
                    AugmentationStrategy.WORD_DELETION,
                    AugmentationStrategy.PUNCTUATION_VARIATION
                ],
                augmentation_factor=1.0,
                max_changes_per_text=5,  # More aggressive changes
                preserve_semantics=False  # Allow more aggressive changes
            )
            
            adversarial = self._augment_single_sample(base_sample, adversarial_config)
            if adversarial:
                adversarial['source'] = 'adversarial'
                adversarial['confidence'] = 0.7  # Lower confidence for adversarial
                adversarial_samples.append(adversarial)
        
        logger.info(f"Created {len(adversarial_samples)} adversarial samples")
        return adversarial_samples