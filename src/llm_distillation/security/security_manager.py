"""
Main security manager coordinating all security features.

This module provides a centralized security management system that
coordinates PII detection, audit logging, API key management, and
compliance monitoring.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import threading

from .pii_detector import PIIDetector, PIIResult, PIIType
from .audit_logger import AuditLogger, AuditEventType, AuditLevel
from .api_key_manager import APIKeyManager
from ..config import Config
from ..exceptions import SecurityError, ErrorCodes

logger = logging.getLogger(__name__)


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    
    # PII detection settings
    enable_pii_detection: bool = True
    pii_strict_mode: bool = True
    pii_risk_threshold: float = 0.7
    
    # Data handling policies
    allow_pii_in_training: bool = False
    require_data_anonymization: bool = True
    max_text_length: int = 10000
    
    # API security
    enable_api_rate_limiting: bool = True
    max_api_calls_per_minute: int = 100
    require_api_key_rotation: bool = True
    api_key_rotation_days: int = 90
    
    # Audit and compliance
    enable_comprehensive_audit: bool = True
    audit_sensitive_operations: bool = True
    retention_days: int = 365
    
    # Access control
    enable_permission_checks: bool = True
    require_explicit_consent: bool = True
    
    # Data export restrictions
    allow_data_export: bool = True
    require_export_approval: bool = False
    max_export_size_mb: int = 100


class SecurityManager:
    """Centralized security management system."""
    
    def __init__(self, config: Config, policy: Optional[SecurityPolicy] = None):
        self.config = config
        self.policy = policy or SecurityPolicy()
        
        # Initialize security components
        self.pii_detector = PIIDetector(
            strict_mode=self.policy.pii_strict_mode
        ) if self.policy.enable_pii_detection else None
        
        self.audit_logger = AuditLogger(config) if self.policy.enable_comprehensive_audit else None
        self.api_key_manager = APIKeyManager(config)
        
        # Security state tracking
        self.security_incidents: List[Dict[str, Any]] = []
        self.api_call_history: List[Tuple[float, str]] = []
        self._lock = threading.Lock()
        
        # Performance tracking
        self.performance_metrics = {
            'pii_scans': 0,
            'pii_detections': 0,
            'api_calls_blocked': 0,
            'security_violations': 0
        }
        
        logger.info("Security Manager initialized")
        
        if self.audit_logger:
            self.audit_logger.log_event(
                AuditEventType.APPLICATION_START,
                "Security Manager initialized",
                component="security_manager",
                data={
                    'policy': self.policy.__dict__,
                    'pii_detection_enabled': self.policy.enable_pii_detection,
                    'audit_enabled': self.policy.enable_comprehensive_audit
                }
            )
    
    def scan_text_for_pii(
        self,
        text: str,
        context: str = "",
        user_id: Optional[str] = None
    ) -> Tuple[PIIResult, bool]:
        """
        Scan text for PII and determine if it's safe to use.
        
        Args:
            text: Text to scan
            context: Context information (e.g., "training_data", "user_input")
            user_id: User identifier for audit trail
            
        Returns:
            Tuple of (PIIResult, is_safe_to_use)
        """
        
        if not self.pii_detector:
            # PII detection disabled - return safe result
            from .pii_detector import PIIResult
            return PIIResult(original_text=text, cleaned_text=text), True
        
        try:
            with self._lock:
                self.performance_metrics['pii_scans'] += 1
            
            # Perform PII detection
            pii_result = self.pii_detector.detect_pii(text)
            
            # Determine safety
            is_safe = self._evaluate_pii_safety(pii_result, context)
            
            # Log detection if PII found
            if pii_result.has_pii:
                with self._lock:
                    self.performance_metrics['pii_detections'] += 1
                
                self._log_pii_detection(pii_result, context, user_id, is_safe)
            
            return pii_result, is_safe
            
        except Exception as e:
            logger.error(f"PII scanning failed: {e}")
            self._log_security_incident("pii_scan_failure", str(e), "high")
            
            # Fail safe - assume unsafe if scanning fails
            from .pii_detector import PIIResult
            return PIIResult(original_text=text, cleaned_text=text), False
    
    def _evaluate_pii_safety(self, pii_result: PIIResult, context: str) -> bool:
        """Evaluate if text with PII is safe to use based on policy."""
        
        if not pii_result.has_pii:
            return True
        
        # Check risk threshold
        if pii_result.risk_score > self.policy.pii_risk_threshold:
            return False
        
        # Context-specific rules
        if context == "training_data" and not self.policy.allow_pii_in_training:
            return False
        
        # Check for high-risk PII types
        high_risk_types = {PIIType.SSN, PIIType.CREDIT_CARD, PIIType.PASSPORT}
        detected_types = {match.pii_type for match in pii_result.matches}
        
        if high_risk_types.intersection(detected_types):
            return False
        
        return True
    
    def _log_pii_detection(
        self,
        pii_result: PIIResult,
        context: str,
        user_id: Optional[str],
        is_safe: bool
    ) -> None:
        """Log PII detection event."""
        
        if not self.audit_logger:
            return
        
        pii_types = [match.pii_type.value for match in pii_result.matches]
        action_taken = "allowed" if is_safe else "blocked"
        
        self.audit_logger.log_pii_detection(
            text_sample=pii_result.original_text[:100],  # First 100 chars for context
            pii_types=pii_types,
            risk_score=pii_result.risk_score,
            action_taken=action_taken
        )
        
        # Log additional context
        self.audit_logger.log_event(
            AuditEventType.SECURITY_VIOLATION if not is_safe else AuditEventType.USER_ACTION,
            f"PII detection: {action_taken}",
            level=AuditLevel.WARNING if not is_safe else AuditLevel.INFO,
            component="security_manager",
            user_id=user_id,
            data={
                'context': context,
                'pii_types': pii_types,
                'risk_score': pii_result.risk_score,
                'num_matches': len(pii_result.matches)
            },
            sensitive_data=True
        )
    
    def validate_api_access(
        self,
        api_name: str,
        user_id: Optional[str] = None,
        required_permission: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate API access based on rate limiting and permissions.
        
        Args:
            api_name: Name of the API being accessed
            user_id: User identifier
            required_permission: Required permission for access
            
        Returns:
            Tuple of (is_allowed, reason_if_denied)
        """
        
        try:
            # Check rate limiting
            if self.policy.enable_api_rate_limiting:
                if not self._check_rate_limit(api_name):
                    with self._lock:
                        self.performance_metrics['api_calls_blocked'] += 1
                    
                    self._log_security_incident(
                        "api_rate_limit_exceeded",
                        f"Rate limit exceeded for API: {api_name}",
                        "medium",
                        user_id=user_id
                    )
                    
                    return False, "Rate limit exceeded"
            
            # Check permissions if required
            if required_permission and self.policy.enable_permission_checks:
                if not self._check_api_permission(api_name, required_permission):
                    return False, f"Missing required permission: {required_permission}"
            
            # Log successful access
            if self.audit_logger:
                self.audit_logger.log_event(
                    AuditEventType.API_CALL,
                    f"API access granted: {api_name}",
                    component="security_manager",
                    user_id=user_id,
                    data={
                        'api_name': api_name,
                        'required_permission': required_permission
                    }
                )
            
            return True, None
            
        except Exception as e:
            logger.error(f"API access validation failed: {e}")
            return False, "Security validation error"
    
    def _check_rate_limit(self, api_name: str) -> bool:
        """Check if API call is within rate limits."""
        
        current_time = time.time()
        minute_ago = current_time - 60
        
        with self._lock:
            # Clean old entries
            self.api_call_history = [
                (timestamp, name) for timestamp, name in self.api_call_history
                if timestamp > minute_ago
            ]
            
            # Count calls for this API in the last minute
            api_calls_count = sum(
                1 for timestamp, name in self.api_call_history
                if name == api_name
            )
            
            # Check limit
            if api_calls_count >= self.policy.max_api_calls_per_minute:
                return False
            
            # Record this call
            self.api_call_history.append((current_time, api_name))
            
        return True
    
    def _check_api_permission(self, api_name: str, required_permission: str) -> bool:
        """Check if API access has required permission."""
        
        # This would integrate with actual permission system
        # For now, implement basic checks
        
        # Check if API key has permission
        api_key_name = f"{api_name}_key"  # Simplified mapping
        
        return self.api_key_manager.validate_key_permissions(
            api_key_name,
            required_permission
        )
    
    def validate_data_export(
        self,
        data_size_mb: float,
        export_type: str,
        user_id: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate data export request.
        
        Args:
            data_size_mb: Size of data to export in MB
            export_type: Type of export (e.g., "dataset", "model")
            user_id: User identifier
            
        Returns:
            Tuple of (is_allowed, reason_if_denied)
        """
        
        if not self.policy.allow_data_export:
            return False, "Data export is disabled by policy"
        
        if data_size_mb > self.policy.max_export_size_mb:
            return False, f"Export size exceeds limit: {data_size_mb:.1f}MB > {self.policy.max_export_size_mb}MB"
        
        if self.policy.require_export_approval:
            # In a real system, this would check for approval
            return False, "Export requires administrative approval"
        
        # Log export
        if self.audit_logger:
            self.audit_logger.log_event(
                AuditEventType.DATA_EXPORT,
                f"Data export approved: {export_type}",
                component="security_manager",
                user_id=user_id,
                data={
                    'export_type': export_type,
                    'size_mb': data_size_mb
                }
            )
        
        return True, None
    
    def _log_security_incident(
        self,
        incident_type: str,
        description: str,
        severity: str,
        user_id: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a security incident."""
        
        incident = {
            'timestamp': time.time(),
            'type': incident_type,
            'description': description,
            'severity': severity,
            'user_id': user_id,
            'data': additional_data or {}
        }
        
        with self._lock:
            self.security_incidents.append(incident)
            self.performance_metrics['security_violations'] += 1
        
        # Log to audit system
        if self.audit_logger:
            self.audit_logger.log_security_violation(
                violation_type=incident_type,
                details=description,
                severity=severity
            )
        
        logger.warning(f"Security incident: {incident_type} - {description}")
    
    def sanitize_text(
        self,
        text: str,
        replacement_strategy: str = "redact"
    ) -> Tuple[str, bool]:
        """
        Sanitize text by removing or replacing PII.
        
        Args:
            text: Text to sanitize
            replacement_strategy: How to handle PII ("redact", "remove", "mask")
            
        Returns:
            Tuple of (sanitized_text, had_pii)
        """
        
        if not self.pii_detector:
            return text, False
        
        pii_result = self.pii_detector.detect_pii(text)
        
        if not pii_result.has_pii:
            return text, False
        
        if replacement_strategy == "redact":
            return pii_result.cleaned_text, True
        elif replacement_strategy == "remove":
            # Remove PII text entirely
            sanitized = text
            for match in reversed(pii_result.matches):  # Reverse to maintain indices
                sanitized = sanitized[:match.start] + sanitized[match.end:]
            return sanitized, True
        elif replacement_strategy == "mask":
            # Replace with generic masks
            sanitized = text
            for match in reversed(pii_result.matches):
                mask = "*" * len(match.text)
                sanitized = sanitized[:match.start] + mask + sanitized[match.end:]
            return sanitized, True
        else:
            return pii_result.cleaned_text, True
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security status and metrics."""
        
        with self._lock:
            metrics = self.performance_metrics.copy()
        
        status = {
            'policy': self.policy.__dict__,
            'performance_metrics': metrics,
            'recent_incidents': self.security_incidents[-10:],  # Last 10 incidents
            'api_key_stats': self.api_key_manager.get_usage_statistics(),
            'rate_limit_status': {
                'recent_calls': len(self.api_call_history),
                'limit_per_minute': self.policy.max_api_calls_per_minute
            }
        }
        
        # Add component status
        status['components'] = {
            'pii_detector': self.pii_detector is not None,
            'audit_logger': self.audit_logger is not None,
            'api_key_manager': True
        }
        
        return status
    
    def run_security_check(self) -> Dict[str, Any]:
        """Run comprehensive security check."""
        
        check_results = {
            'timestamp': time.time(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        # Check PII detection system
        if self.pii_detector:
            try:
                test_result = self.pii_detector.detect_pii("test@example.com")
                check_results['checks']['pii_detection'] = {
                    'status': 'healthy' if test_result.has_pii else 'warning',
                    'message': 'PII detection working' if test_result.has_pii else 'PII detection may not be working'
                }
            except Exception as e:
                check_results['checks']['pii_detection'] = {
                    'status': 'error',
                    'message': f'PII detection error: {e}'
                }
                check_results['overall_status'] = 'error'
        
        # Check API key management
        try:
            api_stats = self.api_key_manager.get_usage_statistics()
            check_results['checks']['api_keys'] = {
                'status': 'healthy',
                'message': f"Managing {api_stats['total_keys']} API keys"
            }
        except Exception as e:
            check_results['checks']['api_keys'] = {
                'status': 'error',
                'message': f'API key management error: {e}'
            }
            check_results['overall_status'] = 'error'
        
        # Check for recent security incidents
        recent_incidents = len([
            incident for incident in self.security_incidents
            if time.time() - incident['timestamp'] < 3600  # Last hour
        ])
        
        if recent_incidents > 10:
            check_results['checks']['incidents'] = {
                'status': 'warning',
                'message': f'{recent_incidents} security incidents in the last hour'
            }
            if check_results['overall_status'] == 'healthy':
                check_results['overall_status'] = 'warning'
        else:
            check_results['checks']['incidents'] = {
                'status': 'healthy',
                'message': f'{recent_incidents} incidents in the last hour'
            }
        
        # Log security check
        if self.audit_logger:
            self.audit_logger.log_event(
                AuditEventType.USER_ACTION,
                "Security check completed",
                component="security_manager",
                data={
                    'overall_status': check_results['overall_status'],
                    'checks_count': len(check_results['checks'])
                }
            )
        
        return check_results
    
    def cleanup_old_data(self, days_to_keep: Optional[int] = None) -> Dict[str, int]:
        """Clean up old security data based on retention policy."""
        
        if days_to_keep is None:
            days_to_keep = self.policy.retention_days
        
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        cleanup_stats = {}
        
        # Clean up incidents
        with self._lock:
            old_incidents = len(self.security_incidents)
            self.security_incidents = [
                incident for incident in self.security_incidents
                if incident['timestamp'] > cutoff_time
            ]
            cleanup_stats['incidents_removed'] = old_incidents - len(self.security_incidents)
        
        # Clean up API call history (keep only last hour)
        hour_ago = time.time() - 3600
        with self._lock:
            old_calls = len(self.api_call_history)
            self.api_call_history = [
                (timestamp, name) for timestamp, name in self.api_call_history
                if timestamp > hour_ago
            ]
            cleanup_stats['api_calls_removed'] = old_calls - len(self.api_call_history)
        
        # Clean up expired API keys
        cleanup_stats['expired_keys_removed'] = self.api_key_manager.cleanup_expired_keys()
        
        logger.info(f"Security cleanup completed: {cleanup_stats}")
        
        return cleanup_stats
    
    def shutdown(self) -> None:
        """Shutdown security manager and cleanup resources."""
        
        if self.audit_logger:
            self.audit_logger.log_event(
                AuditEventType.APPLICATION_STOP,
                "Security Manager shutting down",
                component="security_manager",
                data=self.get_security_status()
            )
            self.audit_logger.close()
        
        # Clear sensitive data from memory
        with self._lock:
            self.security_incidents.clear()
            self.api_call_history.clear()
        
        logger.info("Security Manager shutdown complete")