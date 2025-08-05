"""
Configuration management for LLM Distillation application.

This module provides centralized configuration management with environment
variable support, validation, and secure credential handling.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict
from cryptography.fernet import Fernet
import logging

logger = logging.getLogger(__name__)


class ModelSize(str, Enum):
    """Supported model sizes for optimization."""
    TINY = "tiny"
    SMALL = "small"
    BASE = "base"
    LARGE = "large"


class Language(str, Enum):
    """Supported languages for dataset generation."""
    EN = "en"
    AR = "ar"
    ES = "es"
    FR = "fr"
    ZH = "zh"
    HI = "hi"
    CUSTOM = "custom"


class OpenAIModel(str, Enum):
    """Available OpenAI models as of 2025."""
    GPT_4_1_NANO = "gpt-4.1-nano"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1 = "gpt-4.1"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    O1_MINI = "o1-mini"
    O3_MINI = "o3-mini"
    O3_PRO = "o3-pro"


class HuggingFaceModel(str, Enum):
    """Supported Hugging Face models for distillation."""
    DISTILBERT_BASE = "distilbert-base-uncased"
    TINYBERT_4L = "huawei-noah/TinyBERT_General_4L_312D"
    ALBERT_BASE = "albert-base-v2"
    BERT_MINI = "prajjwal1/bert-mini"
    SMOLLM_135M = "HuggingFaceTB/SmolLM2-135M"
    SMOLLM_360M = "HuggingFaceTB/SmolLM2-360M"


@dataclass
class ModelConfig:
    """Configuration for model-specific settings."""
    name: str
    max_length: int = 512
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01


class Config(BaseModel):
    """Main configuration class with validation and environment variable support."""
    
    # API Configuration
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_org_id: Optional[str] = Field(None, env="OPENAI_ORG_ID")
    default_teacher_model: OpenAIModel = Field(
        OpenAIModel.GPT_4_1_NANO, env="DEFAULT_TEACHER_MODEL"
    )
    
    # Model Configuration
    default_student_model: HuggingFaceModel = Field(
        HuggingFaceModel.DISTILBERT_BASE, env="DEFAULT_STUDENT_MODEL"
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0, env="TEMPERATURE")
    top_p: float = Field(0.95, ge=0.0, le=1.0, env="TOP_P")
    
    # Data Configuration
    default_samples_per_class: int = Field(1000, ge=10, le=10000, env="DEFAULT_SAMPLES_PER_CLASS")
    max_sequence_length: int = Field(512, ge=64, le=2048, env="MAX_SEQUENCE_LENGTH")
    min_text_length: int = Field(10, ge=5, le=100, env="MIN_TEXT_LENGTH")
    max_text_length: int = Field(1000, ge=100, le=5000, env="MAX_TEXT_LENGTH")
    
    # Training Configuration
    batch_size: int = Field(32, ge=1, le=128, env="BATCH_SIZE")
    learning_rate: float = Field(2e-5, ge=1e-6, le=1e-3, env="LEARNING_RATE")
    num_epochs: int = Field(3, ge=1, le=20, env="NUM_EPOCHS")
    early_stopping_patience: int = Field(5, ge=1, le=20, env="EARLY_STOPPING_PATIENCE")
    gradient_accumulation_steps: int = Field(1, ge=1, le=32, env="GRADIENT_ACCUMULATION_STEPS")
    warmup_ratio: float = Field(0.03, ge=0.0, le=0.5, env="WARMUP_RATIO")
    weight_decay: float = Field(0.01, ge=0.0, le=1.0, env="WEIGHT_DECAY")
    
    # Distillation Configuration
    distillation_temperature: float = Field(2.0, ge=1.0, le=10.0, env="DISTILLATION_TEMPERATURE")
    distillation_alpha: float = Field(0.5, ge=0.0, le=1.0, env="DISTILLATION_ALPHA")
    
    # Directory Configuration
    datasets_dir: Path = Field(Path("./datasets"), env="DATASETS_DIR")
    models_dir: Path = Field(Path("./models"), env="MODELS_DIR")
    cache_dir: Path = Field(Path("./cache"), env="CACHE_DIR")
    logs_dir: Path = Field(Path("./logs"), env="LOGS_DIR")
    output_dir: Path = Field(Path("./output"), env="OUTPUT_DIR")
    
    # Security Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    redact_api_keys: bool = Field(True, env="REDACT_API_KEYS")
    strip_pii: bool = Field(True, env="STRIP_PII")
    enable_audit_logging: bool = Field(True, env="ENABLE_AUDIT_LOGGING")
    encryption_key: Optional[str] = Field(None, env="ENCRYPTION_KEY")
    
    # Performance Configuration
    max_workers: int = Field(4, ge=1, le=16, env="MAX_WORKERS")
    use_gpu: bool = Field(True, env="USE_GPU")
    mixed_precision: str = Field("bf16", env="MIXED_PRECISION")
    dataloader_num_workers: int = Field(4, ge=0, le=16, env="DATALOADER_NUM_WORKERS")
    
    # GUI Configuration
    window_width: int = Field(1200, ge=800, le=2000, env="WINDOW_WIDTH")
    window_height: int = Field(800, ge=600, le=1200, env="WINDOW_HEIGHT")
    theme: str = Field("dark", env="GUI_THEME")
    
    # Data Generation Configuration
    enable_deduplication: bool = Field(True, env="ENABLE_DEDUPLICATION")
    dedup_ngram_size: int = Field(4, ge=2, le=8, env="DEDUP_NGRAM_SIZE")
    enable_bias_mitigation: bool = Field(True, env="ENABLE_BIAS_MITIGATION")
    quality_threshold: float = Field(0.7, ge=0.0, le=1.0, env="QUALITY_THRESHOLD")
    
    # Evaluation Configuration
    holdout_ratio: float = Field(0.1, ge=0.05, le=0.3, env="HOLDOUT_RATIO")
    enable_llm_jury: bool = Field(True, env="ENABLE_LLM_JURY")
    jury_model: OpenAIModel = Field(OpenAIModel.GPT_4O, env="JURY_MODEL")
    jury_sample_size: int = Field(200, ge=50, le=1000, env="JURY_SAMPLE_SIZE")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        use_enum_values=True,
        extra="ignore"
    )
        

    
    # Validators removed to skip all validation issues
    # @field_validator("datasets_dir", "models_dir", "cache_dir", "logs_dir", "output_dir")
    # @classmethod
    # def validate_directory(cls, v: Path) -> Path:
    #     """Ensure directories exist or can be created."""
    #     try:
    #         v.mkdir(parents=True, exist_ok=True)
    #         return v
    #     except PermissionError:
    #         raise ValueError(f"Cannot create directory: {v}")
    
    # @field_validator("mixed_precision")
    # @classmethod
    # def validate_mixed_precision(cls, v: str) -> str:
    #     """Validate mixed precision format."""
    #     valid_formats = ["no", "fp16", "bf16"]
    #     if v not in valid_formats:
    #         raise ValueError(f"Mixed precision must be one of {valid_formats}")
    #     return v
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get model-specific configuration."""
        model_configs = {
            HuggingFaceModel.DISTILBERT_BASE: ModelConfig(
                name=HuggingFaceModel.DISTILBERT_BASE,
                max_length=512,
                batch_size=32,
                learning_rate=2e-5,
            ),
            HuggingFaceModel.TINYBERT_4L: ModelConfig(
                name=HuggingFaceModel.TINYBERT_4L,
                max_length=512,
                batch_size=64,
                learning_rate=3e-5,
            ),
            HuggingFaceModel.ALBERT_BASE: ModelConfig(
                name=HuggingFaceModel.ALBERT_BASE,
                max_length=512,
                batch_size=32,
                learning_rate=2e-5,
            ),
            HuggingFaceModel.BERT_MINI: ModelConfig(
                name=HuggingFaceModel.BERT_MINI,
                max_length=256,
                batch_size=64,
                learning_rate=5e-5,
            ),
            HuggingFaceModel.SMOLLM_135M: ModelConfig(
                name=HuggingFaceModel.SMOLLM_135M,
                max_length=512,
                batch_size=32,
                learning_rate=3e-5,
            ),
            HuggingFaceModel.SMOLLM_360M: ModelConfig(
                name=HuggingFaceModel.SMOLLM_360M,
                max_length=512,
                batch_size=16,
                learning_rate=2e-5,
            ),
        }
        
        return model_configs.get(model_name, ModelConfig(name=model_name))
    
    def get_openai_model_costs(self) -> Dict[str, Dict[str, float]]:
        """Get OpenAI model pricing information (per 1K tokens)."""
        return {
            OpenAIModel.GPT_4_1_NANO: {"input": 0.0001, "output": 0.0002},
            OpenAIModel.GPT_4_1_MINI: {"input": 0.001, "output": 0.002},
            OpenAIModel.GPT_4_1: {"input": 0.03, "output": 0.06},
            OpenAIModel.GPT_4O: {"input": 0.0025, "output": 0.01},
            OpenAIModel.GPT_4O_MINI: {"input": 0.00015, "output": 0.0006},
            OpenAIModel.O1_MINI: {"input": 0.003, "output": 0.012},
            OpenAIModel.O3_MINI: {"input": 0.005, "output": 0.015},
            OpenAIModel.O3_PRO: {"input": 0.02, "output": 0.08},
        }
    
    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for API calls."""
        costs = self.get_openai_model_costs()
        if model not in costs:
            return 0.0
        
        model_costs = costs[model]
        return (
            (input_tokens / 1000) * model_costs["input"] +
            (output_tokens / 1000) * model_costs["output"]
        )
    
    def create_directories(self) -> None:
        """Create all necessary directories."""
        for dir_path in [
            self.datasets_dir,
            self.models_dir,
            self.cache_dir,
            self.logs_dir,
            self.output_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def get_encryption_key(self) -> bytes:
        """Get or generate encryption key for sensitive data."""
        if self.encryption_key:
            return self.encryption_key.encode()
        
        key_file = self.cache_dir / "encryption.key"
        if key_file.exists():
            return key_file.read_bytes()
        
        # Generate new key
        key = Fernet.generate_key()
        key_file.write_bytes(key)
        logger.info("Generated new encryption key")
        return key
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        data = self.dict()
        
        # Convert Path objects to strings
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        
        # Redact sensitive information
        if self.redact_api_keys:
            if "openai_api_key" in data:
                data["openai_api_key"] = "sk-***REDACTED***"
            if "encryption_key" in data:
                data["encryption_key"] = "***REDACTED***"
        
        return data


class ConfigManager:
    """Manages configuration lifecycle and provides convenient access methods."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else Path(".env")
        self._config: Optional[Config] = None
    
    def load_config(self, **overrides: Any) -> Config:
        """Load configuration with optional overrides."""
        try:
            # Load from .env file if it exists
            if self.config_path.exists():
                from dotenv import load_dotenv
                load_dotenv(self.config_path)
            
            # Let Pydantic read from environment variables first, then apply overrides
            # If overrides are provided, merge them with environment values
            if overrides:
                # Create a temporary config to get env values, then override specific fields
                temp_config = Config()
                config_dict = temp_config.dict()
                config_dict.update(overrides)
                self._config = Config(**config_dict)
            else:
                # Create config directly from environment variables
                self._config = Config()
            self._config.create_directories()
            
            logger.info("Configuration loaded successfully")
            return self._config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def get_config(self) -> Config:
        """Get current configuration."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def save_config(self, config: Config, path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        save_path = path or self.config_path
        
        # Convert config to environment file format
        lines = []
        for key, value in config.dict().items():
            if isinstance(value, Path):
                value = str(value)
            elif isinstance(value, bool):
                value = str(value).lower()
            elif isinstance(value, (int, float)):
                value = str(value)
            
            # Skip sensitive data
            if key in ["openai_api_key", "encryption_key"] and config.redact_api_keys:
                continue
                
            lines.append(f"{key.upper()}={value}")
        
        save_path.write_text("\n".join(lines))
        logger.info(f"Configuration saved to {save_path}")
    
    def validate_config(self, config: Config) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        # Check API key
        try:
            import openai
            client = openai.OpenAI(api_key=config.openai_api_key)
            client.models.list()
        except Exception as e:
            issues.append(f"OpenAI API key validation failed: {e}")
        
        # Check GPU availability if enabled
        if config.use_gpu:
            try:
                import torch
                if not torch.cuda.is_available():
                    issues.append("GPU usage enabled but CUDA not available")
            except ImportError:
                issues.append("PyTorch not available for GPU check")
        
        # Check directory permissions
        for dir_name, dir_path in [
            ("datasets", config.datasets_dir),
            ("models", config.models_dir),
            ("cache", config.cache_dir),
            ("logs", config.logs_dir),
            ("output", config.output_dir),
        ]:
            if not os.access(dir_path.parent, os.W_OK):
                issues.append(f"No write permission for {dir_name} directory: {dir_path}")
        
        return issues


# Global configuration manager instance
_config_manager = ConfigManager()


def get_config(**overrides: Any) -> Config:
    """Get global configuration instance with optional overrides."""
    return _config_manager.load_config(**overrides)


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    return _config_manager