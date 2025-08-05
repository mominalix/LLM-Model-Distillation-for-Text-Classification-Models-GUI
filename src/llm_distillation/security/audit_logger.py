"""
Audit logging module for compliance and security monitoring.

This module provides comprehensive audit logging capabilities for tracking
user actions, system events, and security-related activities.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib
import uuid

from ..config import Config

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""
    
    # Authentication events
    LOGIN = "login"
    LOGOUT = "logout"
    AUTH_FAILURE = "auth_failure"
    
    # Data events
    DATA_GENERATION_START = "data_generation_start"
    DATA_GENERATION_COMPLETE = "data_generation_complete"
    DATA_GENERATION_FAILED = "data_generation_failed"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    DATA_DELETE = "data_delete"
    
    # Model events
    MODEL_TRAINING_START = "model_training_start"
    MODEL_TRAINING_COMPLETE = "model_training_complete"
    MODEL_TRAINING_FAILED = "model_training_failed"
    MODEL_EXPORT = "model_export"
    MODEL_LOAD = "model_load"
    MODEL_DELETE = "model_delete"
    
    # API events
    API_CALL = "api_call"
    API_ERROR = "api_error"
    API_RATE_LIMIT = "api_rate_limit"
    
    # Security events
    PII_DETECTED = "pii_detected"
    SECURITY_VIOLATION = "security_violation"
    CONFIG_CHANGE = "config_change"
    
    # System events
    APPLICATION_START = "application_start"
    APPLICATION_STOP = "application_stop"
    ERROR = "error"
    WARNING = "warning"
    
    # User actions
    USER_ACTION = "user_action"
    BUTTON_CLICK = "button_click"
    SETTING_CHANGE = "setting_change"


class AuditLevel(str, Enum):
    """Audit event severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Represents an audit event."""
    
    # Core event information
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: AuditEventType = AuditEventType.USER_ACTION
    level: AuditLevel = AuditLevel.INFO
    
    # Event details
    message: str = ""
    description: str = ""
    
    # Context information
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    # System context
    component: str = "application"
    module: str = ""
    function: str = ""
    
    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Security and compliance
    sensitive_data_present: bool = False
    compliance_tags: List[str] = field(default_factory=list)
    
    # Performance metrics
    duration_ms: Optional[float] = None
    resource_usage: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and process the audit event."""
        if not self.message:
            self.message = f"{self.event_type} event"
        
        # Add automatic tags
        self.tags.append(self.event_type.value)
        self.tags.append(self.level.value)
        
        # Add compliance tags based on event type
        if self.event_type in [AuditEventType.PII_DETECTED, AuditEventType.SECURITY_VIOLATION]:
            self.compliance_tags.append("privacy")
        
        if self.event_type in [AuditEventType.DATA_EXPORT, AuditEventType.DATA_IMPORT]:
            self.compliance_tags.append("data_transfer")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    def get_hash(self) -> str:
        """Generate hash for event integrity."""
        event_str = f"{self.event_id}{self.timestamp}{self.event_type}{self.message}"
        return hashlib.sha256(event_str.encode()).hexdigest()


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(self, config: Config):
        self.config = config
        self.enabled = config.enable_audit_logging
        
        # Setup audit log directory
        self.audit_dir = config.logs_dir / "audit"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup log files
        self.current_log_file = self._get_current_log_file()
        self.security_log_file = self.audit_dir / "security.log"
        self.compliance_log_file = self.audit_dir / "compliance.log"
        
        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.events_cache: List[AuditEvent] = []
        self.cache_size = 100
        
        # Statistics
        self.event_counts: Dict[str, int] = {}
        
        if self.enabled:
            logger.info(f"Audit logging initialized - Session: {self.session_id}")
            self.log_event(
                AuditEventType.APPLICATION_START,
                "Audit logging system started",
                component="audit_logger"
            )
    
    def _get_current_log_file(self) -> Path:
        """Get current audit log file (daily rotation)."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.audit_dir / f"audit_{date_str}.log"
    
    def log_event(
        self,
        event_type: AuditEventType,
        message: str,
        level: AuditLevel = AuditLevel.INFO,
        description: str = "",
        component: str = "",
        module: str = "",
        function: str = "",
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        sensitive_data: bool = False
    ) -> AuditEvent:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            message: Brief event message
            level: Event severity level
            description: Detailed description
            component: System component
            module: Module name
            function: Function name
            data: Additional event data
            tags: Custom tags
            user_id: User identifier
            duration_ms: Event duration
            sensitive_data: Whether event contains sensitive data
            
        Returns:
            Created audit event
        """
        
        if not self.enabled:
            return AuditEvent()  # Return empty event if disabled
        
        # Create audit event
        event = AuditEvent(
            event_type=event_type,
            level=level,
            message=message,
            description=description,
            component=component or "application",
            module=module,
            function=function,
            data=data or {},
            tags=tags or [],
            user_id=user_id,
            session_id=self.session_id,
            duration_ms=duration_ms,
            sensitive_data_present=sensitive_data
        )
        
        # Process and store event
        self._process_event(event)
        
        return event
    
    def _process_event(self, event: AuditEvent) -> None:
        """Process and store an audit event."""
        
        # Update statistics
        self.event_counts[event.event_type.value] = self.event_counts.get(event.event_type.value, 0) + 1
        
        # Add to cache
        self.events_cache.append(event)
        if len(self.events_cache) > self.cache_size:
            self.events_cache.pop(0)
        
        # Write to appropriate log files
        self._write_to_log_files(event)
        
        # Special handling for critical events
        if event.level == AuditLevel.CRITICAL:
            self._handle_critical_event(event)
    
    def _write_to_log_files(self, event: AuditEvent) -> None:
        """Write event to appropriate log files."""
        
        # Prepare log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(event.timestamp).isoformat(),
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'level': event.level.value,
            'message': event.message,
            'component': event.component,
            'session_id': event.session_id,
            'hash': event.get_hash()
        }
        
        # Add optional fields
        if event.description:
            log_entry['description'] = event.description
        if event.user_id:
            log_entry['user_id'] = event.user_id
        if event.duration_ms is not None:
            log_entry['duration_ms'] = event.duration_ms
        if event.data:
            # Sanitize sensitive data
            log_entry['data'] = self._sanitize_data(event.data, event.sensitive_data_present)
        
        log_line = json.dumps(log_entry) + '\n'
        
        try:
            # Write to main audit log
            current_file = self._get_current_log_file()
            with open(current_file, 'a', encoding='utf-8') as f:
                f.write(log_line)
            
            # Write to security log for security events
            if event.event_type in [
                AuditEventType.PII_DETECTED,
                AuditEventType.SECURITY_VIOLATION,
                AuditEventType.AUTH_FAILURE
            ]:
                with open(self.security_log_file, 'a', encoding='utf-8') as f:
                    f.write(log_line)
            
            # Write to compliance log for compliance-relevant events
            if event.compliance_tags:
                with open(self.compliance_log_file, 'a', encoding='utf-8') as f:
                    f.write(log_line)
        
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def _sanitize_data(self, data: Dict[str, Any], is_sensitive: bool) -> Dict[str, Any]:
        """Sanitize sensitive data in log entries."""
        
        if not is_sensitive:
            return data
        
        sanitized = {}
        sensitive_keys = {
            'api_key', 'password', 'token', 'secret', 'private_key',
            'ssn', 'credit_card', 'email', 'phone', 'address'
        }
        
        for key, value in data.items():
            key_lower = key.lower()
            
            # Check if key contains sensitive information
            if any(sensitive_key in key_lower for sensitive_key in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 100:
                # Truncate very long strings
                sanitized[key] = value[:100] + "..."
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _handle_critical_event(self, event: AuditEvent) -> None:
        """Handle critical security events."""
        
        # Log to system logger
        logger.critical(f"CRITICAL AUDIT EVENT: {event.message}")
        
        # Could trigger alerts, notifications, etc.
        # This is where you'd integrate with monitoring systems
    
    def log_user_action(
        self,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> AuditEvent:
        """Log a user action."""
        
        return self.log_event(
            AuditEventType.USER_ACTION,
            f"User action: {action}",
            component="gui",
            data=details or {},
            user_id=user_id
        )
    
    def log_api_call(
        self,
        api_name: str,
        method: str,
        status_code: Optional[int] = None,
        duration_ms: Optional[float] = None,
        cost: Optional[float] = None
    ) -> AuditEvent:
        """Log an API call."""
        
        data = {
            'api_name': api_name,
            'method': method
        }
        
        if status_code is not None:
            data['status_code'] = status_code
        if cost is not None:
            data['cost'] = cost
        
        level = AuditLevel.ERROR if status_code and status_code >= 400 else AuditLevel.INFO
        
        return self.log_event(
            AuditEventType.API_CALL,
            f"API call: {api_name}",
            level=level,
            component="api",
            data=data,
            duration_ms=duration_ms
        )
    
    def log_pii_detection(
        self,
        text_sample: str,
        pii_types: List[str],
        risk_score: float,
        action_taken: str = "flagged"
    ) -> AuditEvent:
        """Log PII detection event."""
        
        data = {
            'pii_types': pii_types,
            'risk_score': risk_score,
            'action_taken': action_taken,
            'text_sample_hash': hashlib.sha256(text_sample.encode()).hexdigest()[:16]
        }
        
        level = AuditLevel.WARNING if risk_score > 0.7 else AuditLevel.INFO
        
        return self.log_event(
            AuditEventType.PII_DETECTED,
            f"PII detected: {', '.join(pii_types)}",
            level=level,
            component="security",
            data=data,
            sensitive_data=True,
            tags=["privacy", "compliance"]
        )
    
    def log_data_generation(
        self,
        status: str,
        samples_generated: int = 0,
        duration_ms: Optional[float] = None,
        cost: Optional[float] = None,
        quality_score: Optional[float] = None
    ) -> AuditEvent:
        """Log data generation event."""
        
        event_type_map = {
            'started': AuditEventType.DATA_GENERATION_START,
            'completed': AuditEventType.DATA_GENERATION_COMPLETE,
            'failed': AuditEventType.DATA_GENERATION_FAILED
        }
        
        event_type = event_type_map.get(status, AuditEventType.DATA_GENERATION_START)
        level = AuditLevel.ERROR if status == 'failed' else AuditLevel.INFO
        
        data = {'samples_generated': samples_generated}
        if cost is not None:
            data['cost'] = cost
        if quality_score is not None:
            data['quality_score'] = quality_score
        
        return self.log_event(
            event_type,
            f"Data generation {status}",
            level=level,
            component="data_generator",
            data=data,
            duration_ms=duration_ms
        )
    
    def log_training_event(
        self,
        status: str,
        model_name: str,
        duration_ms: Optional[float] = None,
        final_metrics: Optional[Dict[str, float]] = None
    ) -> AuditEvent:
        """Log model training event."""
        
        event_type_map = {
            'started': AuditEventType.MODEL_TRAINING_START,
            'completed': AuditEventType.MODEL_TRAINING_COMPLETE,
            'failed': AuditEventType.MODEL_TRAINING_FAILED
        }
        
        event_type = event_type_map.get(status, AuditEventType.MODEL_TRAINING_START)
        level = AuditLevel.ERROR if status == 'failed' else AuditLevel.INFO
        
        data = {'model_name': model_name}
        if final_metrics:
            data['final_metrics'] = final_metrics
        
        return self.log_event(
            event_type,
            f"Model training {status}: {model_name}",
            level=level,
            component="trainer",
            data=data,
            duration_ms=duration_ms
        )
    
    def log_security_violation(
        self,
        violation_type: str,
        details: str,
        severity: str = "medium"
    ) -> AuditEvent:
        """Log security violation."""
        
        level_map = {
            'low': AuditLevel.INFO,
            'medium': AuditLevel.WARNING,
            'high': AuditLevel.ERROR,
            'critical': AuditLevel.CRITICAL
        }
        
        level = level_map.get(severity, AuditLevel.WARNING)
        
        return self.log_event(
            AuditEventType.SECURITY_VIOLATION,
            f"Security violation: {violation_type}",
            level=level,
            description=details,
            component="security",
            data={'violation_type': violation_type, 'severity': severity},
            tags=["security", "violation"]
        )
    
    def get_recent_events(self, count: int = 50) -> List[AuditEvent]:
        """Get recent audit events from cache."""
        return self.events_cache[-count:]
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get audit event statistics."""
        
        total_events = sum(self.event_counts.values())
        
        return {
            'session_id': self.session_id,
            'total_events': total_events,
            'event_type_counts': self.event_counts.copy(),
            'cache_size': len(self.events_cache),
            'log_files': {
                'audit': str(self.current_log_file),
                'security': str(self.security_log_file),
                'compliance': str(self.compliance_log_file)
            }
        }
    
    def export_logs(
        self,
        output_path: Path,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_types: Optional[List[AuditEventType]] = None
    ) -> int:
        """
        Export audit logs to file.
        
        Args:
            output_path: Output file path
            start_time: Start timestamp (Unix time)
            end_time: End timestamp (Unix time)
            event_types: Filter by event types
            
        Returns:
            Number of events exported
        """
        
        exported_count = 0
        
        try:
            with open(output_path, 'w', encoding='utf-8') as output_file:
                # Export from current cache
                for event in self.events_cache:
                    if self._should_export_event(event, start_time, end_time, event_types):
                        output_file.write(event.to_json() + '\n')
                        exported_count += 1
                
                # Export from log files if needed
                if start_time or end_time or event_types:
                    exported_count += self._export_from_log_files(
                        output_file, start_time, end_time, event_types
                    )
            
            logger.info(f"Exported {exported_count} audit events to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export audit logs: {e}")
            raise
        
        return exported_count
    
    def _should_export_event(
        self,
        event: AuditEvent,
        start_time: Optional[float],
        end_time: Optional[float],
        event_types: Optional[List[AuditEventType]]
    ) -> bool:
        """Check if event should be exported based on filters."""
        
        if start_time and event.timestamp < start_time:
            return False
        
        if end_time and event.timestamp > end_time:
            return False
        
        if event_types and event.event_type not in event_types:
            return False
        
        return True
    
    def _export_from_log_files(
        self,
        output_file,
        start_time: Optional[float],
        end_time: Optional[float],
        event_types: Optional[List[AuditEventType]]
    ) -> int:
        """Export events from existing log files."""
        
        # This would read from existing log files and filter events
        # For brevity, this is a simplified implementation
        return 0
    
    def close(self) -> None:
        """Close audit logger and perform cleanup."""
        
        if self.enabled:
            self.log_event(
                AuditEventType.APPLICATION_STOP,
                "Audit logging system stopped",
                component="audit_logger"
            )
            
            logger.info(f"Audit logger closed - Total events: {sum(self.event_counts.values())}")


# Context manager for tracking operations
class AuditContext:
    """Context manager for tracking audit events with duration."""
    
    def __init__(
        self,
        audit_logger: AuditLogger,
        event_type: AuditEventType,
        message: str,
        component: str = "",
        **kwargs
    ):
        self.audit_logger = audit_logger
        self.event_type = event_type
        self.message = message
        self.component = component
        self.kwargs = kwargs
        self.start_time = None
        self.event = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        
        level = AuditLevel.ERROR if exc_type else AuditLevel.INFO
        message = f"{self.message} - {'failed' if exc_type else 'completed'}"
        
        self.event = self.audit_logger.log_event(
            self.event_type,
            message,
            level=level,
            component=self.component,
            duration_ms=duration_ms,
            **self.kwargs
        )