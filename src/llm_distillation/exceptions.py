"""
Custom exceptions for LLM Distillation application.

This module defines application-specific exceptions with detailed error
information for better debugging and user experience.
"""

from typing import Any, Dict, Optional


class DistillationError(Exception):
    """Base exception for all distillation-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_error = original_error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "original_error": str(self.original_error) if self.original_error else None,
        }


class ConfigurationError(DistillationError):
    """Raised when configuration is invalid or missing."""
    pass


class DataGenerationError(DistillationError):
    """Raised when data generation fails."""
    pass


class ModelLoadError(DistillationError):
    """Raised when model loading fails."""
    pass


class TrainingError(DistillationError):
    """Raised when training process fails."""
    pass


class EvaluationError(DistillationError):
    """Raised when model evaluation fails."""
    pass


class SecurityError(DistillationError):
    """Raised when security validation fails."""
    pass


class APIError(DistillationError):
    """Raised when external API calls fail."""
    pass


class ValidationError(DistillationError):
    """Raised when data validation fails."""
    pass


class ResourceError(DistillationError):
    """Raised when system resources are insufficient."""
    pass


class GUIError(DistillationError):
    """Raised when GUI operations fail."""
    pass


# Specific error codes for better categorization
class ErrorCodes:
    """Standard error codes for the application."""
    
    # Configuration errors (1000-1099)
    CONFIG_MISSING = "E1001"
    CONFIG_INVALID = "E1002"
    CONFIG_API_KEY_INVALID = "E1003"
    CONFIG_DIRECTORY_PERMISSION = "E1004"
    
    # Data generation errors (1100-1199)
    DATA_GENERATION_FAILED = "E1101"
    DATA_QUALITY_LOW = "E1102"
    DATA_DEDUPLICATION_FAILED = "E1103"
    DATA_PII_DETECTED = "E1104"
    DATA_BIAS_DETECTED = "E1105"
    
    # Model errors (1200-1299)
    MODEL_LOAD_FAILED = "E1201"
    MODEL_DOWNLOAD_FAILED = "E1202"
    MODEL_INCOMPATIBLE = "E1203"
    MODEL_CORRUPTED = "E1204"
    
    # Training errors (1300-1399)
    TRAINING_FAILED = "E1301"
    TRAINING_OOM = "E1302"
    TRAINING_CONVERGENCE_FAILED = "E1303"
    TRAINING_CHECKPOINT_FAILED = "E1304"
    
    # Evaluation errors (1400-1499)
    EVALUATION_FAILED = "E1401"
    EVALUATION_METRICS_INVALID = "E1402"
    EVALUATION_JURY_FAILED = "E1403"
    
    # Security errors (1500-1599)
    SECURITY_API_KEY_EXPOSED = "E1501"
    SECURITY_PII_DETECTED = "E1502"
    SECURITY_UNAUTHORIZED_ACCESS = "E1503"
    
    # API errors (1600-1699)
    API_CONNECTION_FAILED = "E1601"
    API_RATE_LIMIT = "E1602"
    API_QUOTA_EXCEEDED = "E1603"
    API_INVALID_RESPONSE = "E1604"
    
    # Resource errors (1700-1799)
    RESOURCE_INSUFFICIENT_MEMORY = "E1701"
    RESOURCE_INSUFFICIENT_DISK = "E1702"
    RESOURCE_GPU_UNAVAILABLE = "E1703"
    
    # GUI errors (1800-1899)
    GUI_INITIALIZATION_FAILED = "E1801"
    GUI_COMPONENT_ERROR = "E1802"
    GUI_THEME_LOAD_FAILED = "E1803"


def create_error(
    error_class: type,
    message: str,
    error_code: Optional[str] = None,
    **details: Any
) -> DistillationError:
    """Create a standardized error with details."""
    return error_class(
        message=message,
        error_code=error_code,
        details=details
    )


def handle_openai_error(error: Exception) -> APIError:
    """Convert OpenAI API errors to standardized format."""
    error_message = str(error)
    error_code = None
    details = {}
    
    if "rate limit" in error_message.lower():
        error_code = ErrorCodes.API_RATE_LIMIT
        details["retry_after"] = getattr(error, "retry_after", None)
    elif "quota" in error_message.lower():
        error_code = ErrorCodes.API_QUOTA_EXCEEDED
    elif "connection" in error_message.lower():
        error_code = ErrorCodes.API_CONNECTION_FAILED
    else:
        error_code = ErrorCodes.API_INVALID_RESPONSE
    
    return APIError(
        message=f"OpenAI API error: {error_message}",
        error_code=error_code,
        details=details,
        original_error=error
    )


def handle_huggingface_error(error: Exception) -> ModelLoadError:
    """Convert Hugging Face errors to standardized format."""
    error_message = str(error)
    error_code = None
    
    if "connection" in error_message.lower() or "download" in error_message.lower():
        error_code = ErrorCodes.MODEL_DOWNLOAD_FAILED
    elif "corrupted" in error_message.lower() or "invalid" in error_message.lower():
        error_code = ErrorCodes.MODEL_CORRUPTED
    else:
        error_code = ErrorCodes.MODEL_LOAD_FAILED
    
    return ModelLoadError(
        message=f"Hugging Face model error: {error_message}",
        error_code=error_code,
        original_error=error
    )


def handle_torch_error(error: Exception) -> TrainingError:
    """Convert PyTorch errors to standardized format."""
    error_message = str(error)
    error_code = None
    details = {}
    
    if "out of memory" in error_message.lower() or "oom" in error_message.lower():
        error_code = ErrorCodes.TRAINING_OOM
        details["suggestion"] = "Reduce batch size or enable gradient accumulation"
    elif "cuda" in error_message.lower():
        error_code = ErrorCodes.RESOURCE_GPU_UNAVAILABLE
        details["suggestion"] = "Check CUDA installation or use CPU"
    else:
        error_code = ErrorCodes.TRAINING_FAILED
    
    return TrainingError(
        message=f"PyTorch training error: {error_message}",
        error_code=error_code,
        details=details,
        original_error=error
    )


class ErrorHandler:
    """Centralized error handling and logging."""
    
    def __init__(self, logger):
        self.logger = logger
        self.error_counts = {}
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[str] = None,
        reraise: bool = True
    ) -> None:
        """Handle and log errors with context."""
        # Convert to application error if needed
        if isinstance(error, DistillationError):
            app_error = error
        elif "openai" in str(type(error)).lower():
            app_error = handle_openai_error(error)
        elif "transformers" in str(type(error)).lower():
            app_error = handle_huggingface_error(error)
        elif "torch" in str(type(error)).lower():
            app_error = handle_torch_error(error)
        else:
            app_error = DistillationError(
                message=str(error),
                original_error=error
            )
        
        # Track error frequency
        error_key = f"{app_error.__class__.__name__}:{app_error.error_code}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log error with context
        log_data = app_error.to_dict()
        if context:
            log_data["context"] = context
        log_data["occurrence_count"] = self.error_counts[error_key]
        
        self.logger.error(
            f"Error in {context or 'unknown context'}: {app_error.message}",
            extra=log_data
        )
        
        if reraise:
            raise app_error
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error occurrences."""
        return self.error_counts.copy()
    
    def reset_error_counts(self) -> None:
        """Reset error count tracking."""
        self.error_counts.clear()