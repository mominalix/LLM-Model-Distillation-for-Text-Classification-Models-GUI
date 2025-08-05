"""
PII (Personally Identifiable Information) Detection Module.

This module provides comprehensive PII detection capabilities to ensure
data privacy and compliance with regulations like GDPR.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class PIIType(str, Enum):
    """Types of PII that can be detected."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    URL = "url"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BANK_ACCOUNT = "bank_account"
    CUSTOM = "custom"


@dataclass
class PIIMatch:
    """Represents a PII match found in text."""
    pii_type: PIIType
    text: str
    start: int
    end: int
    confidence: float
    context: str = ""
    suggested_replacement: str = "[REDACTED]"


@dataclass
class PIIResult:
    """Result of PII detection on a text."""
    original_text: str
    matches: List[PIIMatch] = field(default_factory=list)
    cleaned_text: str = ""
    has_pii: bool = False
    risk_score: float = 0.0
    
    def __post_init__(self):
        """Calculate derived fields."""
        self.has_pii = len(self.matches) > 0
        if self.matches:
            # Calculate risk score based on number and types of PII
            risk_scores = {
                PIIType.SSN: 1.0,
                PIIType.CREDIT_CARD: 1.0,
                PIIType.PASSPORT: 0.9,
                PIIType.DRIVER_LICENSE: 0.8,
                PIIType.BANK_ACCOUNT: 0.9,
                PIIType.EMAIL: 0.6,
                PIIType.PHONE: 0.7,
                PIIType.NAME: 0.5,
                PIIType.ADDRESS: 0.8,
                PIIType.DATE_OF_BIRTH: 0.7,
                PIIType.IP_ADDRESS: 0.3,
                PIIType.URL: 0.2,
            }
            
            max_risk = max(risk_scores.get(match.pii_type, 0.5) for match in self.matches)
            num_matches_factor = min(len(self.matches) * 0.1, 0.5)
            self.risk_score = min(max_risk + num_matches_factor, 1.0)


class PIIDetector:
    """Advanced PII detection with multiple detection strategies."""
    
    def __init__(self, strict_mode: bool = True, custom_patterns: Optional[Dict[str, str]] = None):
        """
        Initialize PII detector.
        
        Args:
            strict_mode: Whether to use strict detection (more false positives, fewer false negatives)
            custom_patterns: Custom regex patterns for additional PII types
        """
        self.strict_mode = strict_mode
        self.custom_patterns = custom_patterns or {}
        
        # Compiled regex patterns for performance
        self.patterns: Dict[PIIType, List[Pattern]] = self._compile_patterns()
        
        # Common words that should not be flagged as names
        self.name_exceptions = self._load_name_exceptions()
        
        # Statistical models for name detection (simplified)
        self.common_first_names = self._load_common_names("first")
        self.common_last_names = self._load_common_names("last")
        
        logger.info(f"PII Detector initialized with strict_mode={strict_mode}")
    
    def _compile_patterns(self) -> Dict[PIIType, List[Pattern]]:
        """Compile regex patterns for PII detection."""
        
        patterns = {}
        
        # Email patterns
        patterns[PIIType.EMAIL] = [
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE),
            re.compile(r'\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\s*\.\s*[A-Z|a-z]{2,}\b', re.IGNORECASE),
        ]
        
        # Phone number patterns (various formats)
        patterns[PIIType.PHONE] = [
            re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            re.compile(r'\b(?:\+?1[-.\s]?)?[0-9]{3}[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            re.compile(r'\b(?:\+?[1-9][0-9]{0,3}[-.\s]?)?[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}\b'),
        ]
        
        # Social Security Number
        patterns[PIIType.SSN] = [
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            re.compile(r'\b\d{3}\s\d{2}\s\d{4}\b'),
            re.compile(r'\b\d{9}\b'),  # 9 consecutive digits
        ]
        
        # Credit card numbers
        patterns[PIIType.CREDIT_CARD] = [
            re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),  # 16 digits with optional separators
            re.compile(r'\b(?:\d{4}[-\s]?){2}\d{4}[-\s]?\d{3}\b'),  # 15 digits (Amex)
            re.compile(r'\b\d{13,19}\b'),  # 13-19 consecutive digits
        ]
        
        # IP addresses
        patterns[PIIType.IP_ADDRESS] = [
            re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),  # IPv6
        ]
        
        # URLs
        patterns[PIIType.URL] = [
            re.compile(r'https?://[^\s]+', re.IGNORECASE),
            re.compile(r'www\.[^\s]+', re.IGNORECASE),
            re.compile(r'\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?\b'),
        ]
        
        # Names (basic pattern)
        patterns[PIIType.NAME] = [
            re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),  # First Last
            re.compile(r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b'),  # First M. Last
            re.compile(r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.? [A-Z][a-z]+ [A-Z][a-z]+\b'),  # Title First Last
        ]
        
        # Addresses (simplified)
        patterns[PIIType.ADDRESS] = [
            re.compile(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Boulevard|Blvd)\b', re.IGNORECASE),
            re.compile(r'\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr)\b'),
        ]
        
        # Date of birth patterns
        patterns[PIIType.DATE_OF_BIRTH] = [
            re.compile(r'\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}\b'),  # MM/DD/YYYY
            re.compile(r'\b(?:19|20)\d{2}[/-](?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])\b'),  # YYYY/MM/DD
            re.compile(r'\b(?:0?[1-9]|[12][0-9]|3[01])[/-](?:0?[1-9]|1[0-2])[/-](?:19|20)\d{2}\b'),  # DD/MM/YYYY
        ]
        
        # Passport numbers (simplified)
        patterns[PIIType.PASSPORT] = [
            re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'),  # Basic passport format
        ]
        
        # Driver's license (simplified)
        patterns[PIIType.DRIVER_LICENSE] = [
            re.compile(r'\b[A-Z]\d{7}\b'),  # Some state formats
            re.compile(r'\b\d{8,12}\b'),  # Numeric formats
        ]
        
        # Bank account numbers (simplified)
        patterns[PIIType.BANK_ACCOUNT] = [
            re.compile(r'\b\d{8,17}\b'),  # 8-17 digit account numbers
        ]
        
        # Add custom patterns
        for pii_type, pattern in self.custom_patterns.items():
            try:
                if PIIType.CUSTOM not in patterns:
                    patterns[PIIType.CUSTOM] = []
                patterns[PIIType.CUSTOM].append(re.compile(pattern))
            except re.error as e:
                logger.warning(f"Invalid custom pattern for {pii_type}: {e}")
        
        return patterns
    
    def _load_name_exceptions(self) -> Set[str]:
        """Load common words that should not be flagged as names."""
        exceptions = {
            # Common words that look like names but aren't
            'The', 'This', 'That', 'They', 'There', 'Then', 'Thank', 'Thanks',
            'Hello', 'Hi', 'Hey', 'Yes', 'No', 'Maybe', 'Please', 'Sorry',
            'Welcome', 'Good', 'Great', 'Best', 'Better', 'Nice', 'Fine',
            'Here', 'Where', 'When', 'What', 'Who', 'Why', 'How',
            'First', 'Last', 'Next', 'Previous', 'Current', 'New', 'Old',
            'Big', 'Small', 'Large', 'Little', 'High', 'Low', 'Fast', 'Slow',
            # Days and months
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December',
            # Company/brand names that look like person names
            'Apple', 'Amazon', 'Google', 'Microsoft', 'Facebook', 'Twitter',
        }
        return exceptions
    
    def _load_common_names(self, name_type: str) -> Set[str]:
        """Load common first/last names for statistical detection."""
        # In a production system, these would be loaded from comprehensive databases
        
        if name_type == "first":
            return {
                'James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Thomas',
                'Christopher', 'Daniel', 'Matthew', 'Anthony', 'Mark', 'Donald', 'Steven',
                'Paul', 'Andrew', 'Joshua', 'Kenneth', 'Kevin', 'Brian', 'George', 'Edward',
                'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan',
                'Jessica', 'Sarah', 'Karen', 'Nancy', 'Lisa', 'Betty', 'Dorothy', 'Sandra',
                'Ashley', 'Kimberly', 'Emily', 'Donna', 'Margaret', 'Carol', 'Michelle',
            }
        else:  # last names
            return {
                'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
                'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
                'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
                'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark',
                'Ramirez', 'Lewis', 'Robinson', 'Walker', 'Young', 'Allen', 'King',
            }
    
    def detect_pii(self, text: str) -> PIIResult:
        """
        Detect PII in the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            PIIResult with detected PII information
        """
        matches = []
        
        # Apply pattern-based detection
        for pii_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Additional validation for certain PII types
                    if self._validate_match(pii_type, match.group(), text):
                        confidence = self._calculate_confidence(pii_type, match.group(), text)
                        
                        if confidence >= (0.7 if self.strict_mode else 0.8):
                            pii_match = PIIMatch(
                                pii_type=pii_type,
                                text=match.group(),
                                start=match.start(),
                                end=match.end(),
                                confidence=confidence,
                                context=self._extract_context(text, match.start(), match.end()),
                                suggested_replacement=self._get_replacement(pii_type)
                            )
                            matches.append(pii_match)
        
        # Remove duplicates and overlapping matches
        matches = self._deduplicate_matches(matches)
        
        # Generate cleaned text
        cleaned_text = self._generate_cleaned_text(text, matches)
        
        return PIIResult(
            original_text=text,
            matches=matches,
            cleaned_text=cleaned_text
        )
    
    def _validate_match(self, pii_type: PIIType, match_text: str, full_text: str) -> bool:
        """Validate a potential PII match with additional checks."""
        
        if pii_type == PIIType.NAME:
            # Check if it's in our exceptions list
            words = match_text.split()
            for word in words:
                if word in self.name_exceptions:
                    return False
            
            # Check if it's likely a real name using statistical methods
            return self._is_likely_name(match_text)
        
        elif pii_type == PIIType.CREDIT_CARD:
            # Validate using Luhn algorithm
            return self._is_valid_credit_card(match_text)
        
        elif pii_type == PIIType.EMAIL:
            # Additional email validation
            return self._is_valid_email(match_text)
        
        elif pii_type == PIIType.SSN:
            # Validate SSN format and known invalid patterns
            return self._is_valid_ssn(match_text)
        
        elif pii_type == PIIType.PHONE:
            # Validate phone number format
            return self._is_valid_phone(match_text)
        
        return True
    
    def _is_likely_name(self, text: str) -> bool:
        """Check if text is likely to be a real name."""
        words = text.split()
        
        if len(words) != 2:  # For now, only check first-last name patterns
            return False
        
        first_name, last_name = words
        
        # Check against common name databases
        first_is_common = first_name in self.common_first_names
        last_is_common = last_name in self.common_last_names
        
        # In strict mode, require at least one common name
        if self.strict_mode:
            return first_is_common or last_is_common
        else:
            return first_is_common and last_is_common
    
    def _is_valid_credit_card(self, text: str) -> bool:
        """Validate credit card using Luhn algorithm."""
        # Remove non-digits
        digits = re.sub(r'\D', '', text)
        
        if len(digits) < 13 or len(digits) > 19:
            return False
        
        # Luhn algorithm
        def luhn_checksum(digits):
            def digits_of(n):
                return [int(d) for d in str(n)]
            
            digits = digits_of(digits)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d*2))
            return checksum % 10
        
        return luhn_checksum(digits) == 0
    
    def _is_valid_email(self, email: str) -> bool:
        """Additional email validation."""
        # Basic checks
        if email.count('@') != 1:
            return False
        
        local, domain = email.split('@')
        
        # Check for valid domain
        if '.' not in domain:
            return False
        
        # Check for reasonable length
        if len(local) > 64 or len(domain) > 253:
            return False
        
        return True
    
    def _is_valid_ssn(self, ssn: str) -> bool:
        """Validate SSN format."""
        # Remove non-digits
        digits = re.sub(r'\D', '', ssn)
        
        if len(digits) != 9:
            return False
        
        # Check for known invalid patterns
        invalid_patterns = [
            '000000000', '111111111', '222222222', '333333333',
            '444444444', '555555555', '666666666', '777777777',
            '888888888', '999999999', '123456789'
        ]
        
        return digits not in invalid_patterns
    
    def _is_valid_phone(self, phone: str) -> bool:
        """Validate phone number."""
        # Remove non-digits
        digits = re.sub(r'\D', '', phone)
        
        # Check length (US: 10-11 digits, international: 7-15)
        if len(digits) < 7 or len(digits) > 15:
            return False
        
        # Check for obviously fake numbers
        if len(set(digits)) <= 2:  # All same digits or only 2 different digits
            return False
        
        return True
    
    def _calculate_confidence(self, pii_type: PIIType, match_text: str, full_text: str) -> float:
        """Calculate confidence score for a PII match."""
        
        base_confidence = {
            PIIType.EMAIL: 0.9,
            PIIType.SSN: 0.95,
            PIIType.CREDIT_CARD: 0.9,
            PIIType.PHONE: 0.8,
            PIIType.IP_ADDRESS: 0.95,
            PIIType.URL: 0.9,
            PIIType.NAME: 0.6,  # Lower due to ambiguity
            PIIType.ADDRESS: 0.7,
            PIIType.DATE_OF_BIRTH: 0.8,
            PIIType.PASSPORT: 0.8,
            PIIType.DRIVER_LICENSE: 0.7,
            PIIType.BANK_ACCOUNT: 0.7,
        }.get(pii_type, 0.7)
        
        # Adjust based on context
        context = self._extract_context(full_text, 0, len(match_text))
        
        # Look for PII-related keywords that increase confidence
        pii_keywords = {
            PIIType.EMAIL: ['email', 'mail', 'contact', '@'],
            PIIType.PHONE: ['phone', 'call', 'number', 'tel'],
            PIIType.SSN: ['ssn', 'social', 'security'],
            PIIType.NAME: ['name', 'person', 'user', 'author'],
            PIIType.ADDRESS: ['address', 'street', 'location'],
        }
        
        keywords = pii_keywords.get(pii_type, [])
        context_lower = context.lower()
        
        keyword_boost = sum(0.05 for keyword in keywords if keyword in context_lower)
        
        return min(base_confidence + keyword_boost, 1.0)
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Extract context around a PII match."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _get_replacement(self, pii_type: PIIType) -> str:
        """Get appropriate replacement text for PII type."""
        replacements = {
            PIIType.EMAIL: "[EMAIL]",
            PIIType.PHONE: "[PHONE]",
            PIIType.SSN: "[SSN]",
            PIIType.CREDIT_CARD: "[CREDIT_CARD]",
            PIIType.IP_ADDRESS: "[IP_ADDRESS]",
            PIIType.URL: "[URL]",
            PIIType.NAME: "[NAME]",
            PIIType.ADDRESS: "[ADDRESS]",
            PIIType.DATE_OF_BIRTH: "[DATE_OF_BIRTH]",
            PIIType.PASSPORT: "[PASSPORT]",
            PIIType.DRIVER_LICENSE: "[DRIVER_LICENSE]",
            PIIType.BANK_ACCOUNT: "[BANK_ACCOUNT]",
        }
        return replacements.get(pii_type, "[REDACTED]")
    
    def _deduplicate_matches(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove duplicate and overlapping matches."""
        if not matches:
            return matches
        
        # Sort by start position
        matches.sort(key=lambda m: m.start)
        
        deduplicated = []
        for match in matches:
            # Check for overlaps with existing matches
            overlaps = False
            for existing in deduplicated:
                if (match.start < existing.end and match.end > existing.start):
                    # Overlapping - keep the one with higher confidence
                    if match.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(match)
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(match)
        
        return sorted(deduplicated, key=lambda m: m.start)
    
    def _generate_cleaned_text(self, text: str, matches: List[PIIMatch]) -> str:
        """Generate text with PII replaced."""
        if not matches:
            return text
        
        # Sort matches by start position in reverse order to maintain indices
        sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)
        
        cleaned = text
        for match in sorted_matches:
            cleaned = (
                cleaned[:match.start] + 
                match.suggested_replacement + 
                cleaned[match.end:]
            )
        
        return cleaned
    
    def batch_detect(self, texts: List[str]) -> List[PIIResult]:
        """Detect PII in multiple texts."""
        return [self.detect_pii(text) for text in texts]
    
    def get_detection_summary(self, results: List[PIIResult]) -> Dict[str, Any]:
        """Get summary statistics for batch PII detection."""
        
        total_texts = len(results)
        texts_with_pii = sum(1 for result in results if result.has_pii)
        
        pii_type_counts = {}
        total_matches = 0
        
        for result in results:
            total_matches += len(result.matches)
            for match in result.matches:
                pii_type_counts[match.pii_type] = pii_type_counts.get(match.pii_type, 0) + 1
        
        avg_risk_score = sum(result.risk_score for result in results) / total_texts if total_texts > 0 else 0
        
        return {
            'total_texts': total_texts,
            'texts_with_pii': texts_with_pii,
            'pii_detection_rate': texts_with_pii / total_texts if total_texts > 0 else 0,
            'total_pii_matches': total_matches,
            'pii_type_counts': pii_type_counts,
            'average_risk_score': avg_risk_score,
            'high_risk_texts': sum(1 for result in results if result.risk_score > 0.7)
        }