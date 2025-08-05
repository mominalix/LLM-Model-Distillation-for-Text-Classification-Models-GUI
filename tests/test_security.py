"""
Tests for security components.

This module tests PII detection, audit logging, API key management,
and overall security functionality.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from llm_distillation.security import PIIDetector, PIIResult, PIIType, AuditLogger, AuditEventType, SecurityManager
from llm_distillation.config import Config


class TestPIIDetector:
    """Test PII detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = PIIDetector(strict_mode=True)
        self.non_strict_detector = PIIDetector(strict_mode=False)
    
    def test_email_detection(self):
        """Test email address detection."""
        text = "Contact me at john.doe@example.com for more information."
        result = self.detector.detect_pii(text)
        
        assert result.has_pii
        assert len(result.matches) == 1
        assert result.matches[0].pii_type == PIIType.EMAIL
        assert result.matches[0].text == "john.doe@example.com"
        assert result.matches[0].confidence > 0.8
    
    def test_phone_detection(self):
        """Test phone number detection."""
        text = "Call me at (555) 123-4567 or 555.987.6543"
        result = self.detector.detect_pii(text)
        
        assert result.has_pii
        phone_matches = [m for m in result.matches if m.pii_type == PIIType.PHONE]
        assert len(phone_matches) == 2
    
    def test_ssn_detection(self):
        """Test Social Security Number detection."""
        text = "My SSN is 123-45-6789"
        result = self.detector.detect_pii(text)
        
        assert result.has_pii
        ssn_matches = [m for m in result.matches if m.pii_type == PIIType.SSN]
        assert len(ssn_matches) == 1
        assert ssn_matches[0].text == "123-45-6789"
    
    def test_credit_card_detection(self):
        """Test credit card number detection."""
        # Valid credit card number (test number)
        text = "Pay with card 4532015112830366"
        result = self.detector.detect_pii(text)
        
        assert result.has_pii
        cc_matches = [m for m in result.matches if m.pii_type == PIIType.CREDIT_CARD]
        assert len(cc_matches) >= 1
    
    def test_name_detection_strict_vs_non_strict(self):
        """Test name detection in strict vs non-strict mode."""
        text = "John Smith is a great developer."
        
        # Strict mode - should detect names more aggressively
        strict_result = self.detector.detect_pii(text)
        
        # Non-strict mode - should be more conservative
        non_strict_result = self.non_strict_detector.detect_pii(text)
        
        # Both should detect the name, but with different confidence
        assert strict_result.has_pii
        name_matches_strict = [m for m in strict_result.matches if m.pii_type == PIIType.NAME]
        assert len(name_matches_strict) >= 1
    
    def test_no_pii_text(self):
        """Test text with no PII."""
        text = "This is a normal sentence with no personal information."
        result = self.detector.detect_pii(text)
        
        assert not result.has_pii
        assert len(result.matches) == 0
        assert result.risk_score == 0.0
    
    def test_cleaned_text_generation(self):
        """Test generation of cleaned text with PII removed."""
        text = "Contact john.doe@example.com or call 555-1234"
        result = self.detector.detect_pii(text)
        
        assert result.has_pii
        assert "[EMAIL]" in result.cleaned_text
        assert "john.doe@example.com" not in result.cleaned_text
    
    def test_risk_score_calculation(self):
        """Test risk score calculation."""
        # High risk text with SSN
        high_risk_text = "My SSN is 123-45-6789 and card is 4532015112830366"
        high_risk_result = self.detector.detect_pii(high_risk_text)
        
        # Low risk text with just email
        low_risk_text = "Contact me at test@example.com"
        low_risk_result = self.detector.detect_pii(low_risk_text)
        
        assert high_risk_result.risk_score > low_risk_result.risk_score
        assert high_risk_result.risk_score > 0.7  # Should be high risk
    
    def test_batch_detection(self):
        """Test batch PII detection."""
        texts = [
            "Contact john@example.com",
            "Call 555-1234",
            "No PII here",
            "SSN: 123-45-6789"
        ]
        
        results = self.detector.batch_detect(texts)
        
        assert len(results) == 4
        pii_count = sum(1 for result in results if result.has_pii)
        assert pii_count == 3  # First three have PII


class TestAuditLogger:
    """Test audit logging functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config(
            openai_api_key="sk-test123456789",
            logs_dir=Path(self.temp_dir),
            enable_audit_logging=True
        )
        self.audit_logger = AuditLogger(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_event_logging(self):
        """Test basic event logging functionality."""
        event = self.audit_logger.log_event(
            AuditEventType.USER_ACTION,
            "Test event",
            component="test",
            data={"test_key": "test_value"}
        )
        
        assert event.event_type == AuditEventType.USER_ACTION
        assert event.message == "Test event"
        assert event.component == "test"
        assert event.data["test_key"] == "test_value"
        assert event.session_id is not None
    
    def test_pii_detection_logging(self):
        """Test PII detection event logging."""
        event = self.audit_logger.log_pii_detection(
            text_sample="Contact john@example.com",
            pii_types=["email"],
            risk_score=0.6,
            action_taken="flagged"
        )
        
        assert event.event_type == AuditEventType.PII_DETECTED
        assert event.sensitive_data_present is True
        assert "email" in event.data["pii_types"]
    
    def test_api_call_logging(self):
        """Test API call logging."""
        event = self.audit_logger.log_api_call(
            api_name="openai",
            method="POST",
            status_code=200,
            duration_ms=1500.0,
            cost=0.01
        )
        
        assert event.event_type == AuditEventType.API_CALL
        assert event.data["api_name"] == "openai"
        assert event.data["status_code"] == 200
        assert event.duration_ms == 1500.0
    
    def test_log_file_creation(self):
        """Test that log files are created properly."""
        self.audit_logger.log_event(
            AuditEventType.USER_ACTION,
            "Test for file creation"
        )
        
        # Check that audit log file exists
        audit_files = list(self.audit_logger.audit_dir.glob("audit_*.log"))
        assert len(audit_files) > 0
        
        # Check file content
        audit_file = audit_files[0]
        content = audit_file.read_text()
        assert "Test for file creation" in content
    
    def test_event_statistics(self):
        """Test event statistics tracking."""
        # Log multiple events
        for i in range(5):
            self.audit_logger.log_event(
                AuditEventType.USER_ACTION,
                f"Test event {i}"
            )
        
        stats = self.audit_logger.get_event_statistics()
        
        assert stats["total_events"] == 6  # 5 + 1 from initialization
        assert AuditEventType.USER_ACTION.value in stats["event_type_counts"]
        assert stats["event_type_counts"][AuditEventType.USER_ACTION.value] == 5


class TestSecurityManager:
    """Test security manager integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config(
            openai_api_key="sk-test123456789",
            logs_dir=Path(self.temp_dir),
            cache_dir=Path(self.temp_dir),
            enable_audit_logging=True
        )
        self.security_manager = SecurityManager(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pii_scanning_integration(self):
        """Test PII scanning through security manager."""
        text_with_pii = "Contact me at john@example.com"
        text_without_pii = "This is safe text"
        
        # Scan text with PII
        pii_result, is_safe = self.security_manager.scan_text_for_pii(
            text_with_pii,
            context="test",
            user_id="test_user"
        )
        
        assert pii_result.has_pii
        # Should be safe for low-risk PII like email in test context
        
        # Scan text without PII
        safe_result, is_safe = self.security_manager.scan_text_for_pii(text_without_pii)
        
        assert not safe_result.has_pii
        assert is_safe
    
    def test_api_access_validation(self):
        """Test API access validation with rate limiting."""
        api_name = "test_api"
        
        # First call should succeed
        allowed, reason = self.security_manager.validate_api_access(api_name)
        assert allowed
        assert reason is None
        
        # Test rate limiting by making many calls
        for _ in range(101):  # Exceed default limit of 100
            self.security_manager.validate_api_access(api_name)
        
        # Should now be rate limited
        allowed, reason = self.security_manager.validate_api_access(api_name)
        assert not allowed
        assert "Rate limit" in reason
    
    def test_data_export_validation(self):
        """Test data export validation."""
        # Small export should be allowed
        allowed, reason = self.security_manager.validate_data_export(
            data_size_mb=10.0,
            export_type="dataset",
            user_id="test_user"
        )
        assert allowed
        assert reason is None
        
        # Large export should be blocked
        allowed, reason = self.security_manager.validate_data_export(
            data_size_mb=200.0,  # Exceeds default 100MB limit
            export_type="dataset"
        )
        assert not allowed
        assert "exceeds limit" in reason
    
    def test_text_sanitization(self):
        """Test text sanitization functionality."""
        text = "Contact john@example.com or call 555-1234"
        
        # Test redaction strategy
        sanitized, had_pii = self.security_manager.sanitize_text(text, "redact")
        assert had_pii
        assert "[EMAIL]" in sanitized
        assert "john@example.com" not in sanitized
        
        # Test removal strategy
        removed, had_pii = self.security_manager.sanitize_text(text, "remove")
        assert had_pii
        assert len(removed) < len(text)
    
    def test_security_status_reporting(self):
        """Test security status reporting."""
        # Generate some activity
        self.security_manager.scan_text_for_pii("test@example.com")
        self.security_manager.validate_api_access("test_api")
        
        status = self.security_manager.get_security_status()
        
        assert "policy" in status
        assert "performance_metrics" in status
        assert "components" in status
        assert status["performance_metrics"]["pii_scans"] > 0
    
    def test_security_check(self):
        """Test comprehensive security check."""
        check_results = self.security_manager.run_security_check()
        
        assert "overall_status" in check_results
        assert "checks" in check_results
        assert check_results["overall_status"] in ["healthy", "warning", "error"]
        
        # Should have checks for main components
        assert "pii_detection" in check_results["checks"]
        assert "api_keys" in check_results["checks"]


class TestSecurityIntegration:
    """Integration tests for security components."""
    
    def test_full_security_workflow(self):
        """Test complete security workflow."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Config(
                openai_api_key="sk-test123456789",
                logs_dir=Path(tmp_dir),
                cache_dir=Path(tmp_dir),
                enable_audit_logging=True
            )
            
            security_manager = SecurityManager(config)
            
            # Simulate data generation with PII detection
            training_text = "User feedback: Contact support at help@company.com"
            pii_result, is_safe = security_manager.scan_text_for_pii(
                training_text,
                context="training_data",
                user_id="data_scientist"
            )
            
            # Validate API access for model training
            api_allowed, _ = security_manager.validate_api_access(
                "openai",
                user_id="data_scientist"
            )
            
            # Validate data export
            export_allowed, _ = security_manager.validate_data_export(
                data_size_mb=5.0,
                export_type="model",
                user_id="data_scientist"
            )
            
            # Get final security status
            final_status = security_manager.get_security_status()
            
            # Assertions
            assert pii_result.has_pii  # Should detect email
            assert api_allowed  # Should allow API access
            assert export_allowed  # Should allow small export
            assert final_status["performance_metrics"]["pii_scans"] > 0


if __name__ == "__main__":
    pytest.main([__file__])