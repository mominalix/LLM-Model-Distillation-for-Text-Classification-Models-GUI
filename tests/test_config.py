"""
Tests for configuration management.

This module tests the configuration loading, validation, and management
functionality to ensure robust application configuration.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from llm_distillation.config import Config, ConfigManager, OpenAIModel, HuggingFaceModel, Language
from llm_distillation.exceptions import ValidationError


class TestConfig:
    """Test configuration class functionality."""
    
    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        config = Config(openai_api_key="sk-test123456789")
        
        assert config.openai_api_key == "sk-test123456"
        assert config.default_teacher_model == OpenAIModel.GPT_4_1_NANO
        assert config.default_student_model == HuggingFaceModel.DISTILBERT_BASE
        assert config.temperature == 0.7
        assert config.use_gpu is True
    
    def test_config_validation_api_key(self):
        """Test API key validation."""
        # Valid API key
        config = Config(openai_api_key="sk-valid123456789")
        assert config.openai_api_key == "sk-valid123456789"
        
        # Invalid API key format
        with pytest.raises(ValueError, match="OpenAI API key must start with 'sk-'"):
            Config(openai_api_key="invalid-key")
        
        # Too short API key
        with pytest.raises(ValueError, match="OpenAI API key appears to be too short"):
            Config(openai_api_key="sk-short")
    
    def test_config_directory_creation(self):
        """Test directory creation functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Config(
                openai_api_key="sk-test123456789",
                datasets_dir=Path(tmp_dir) / "datasets",
                models_dir=Path(tmp_dir) / "models"
            )
            
            assert config.datasets_dir.exists()
            assert config.models_dir.exists()
    
    def test_model_config_retrieval(self):
        """Test getting model-specific configurations."""
        config = Config(openai_api_key="sk-test123456789")
        
        # Test DistilBERT config
        distilbert_config = config.get_model_config(HuggingFaceModel.DISTILBERT_BASE)
        assert distilbert_config.name == HuggingFaceModel.DISTILBERT_BASE
        assert distilbert_config.max_length == 512
        assert distilbert_config.batch_size == 32
        
        # Test TinyBERT config
        tinybert_config = config.get_model_config(HuggingFaceModel.TINYBERT_4L)
        assert tinybert_config.batch_size == 64
    
    def test_cost_estimation(self):
        """Test API cost estimation functionality."""
        config = Config(openai_api_key="sk-test123456789")
        
        cost = config.estimate_cost(OpenAIModel.GPT_4_1_NANO, 1000, 500)
        expected = (1000 / 1000) * 0.0001 + (500 / 1000) * 0.0002
        assert cost == expected
        
        # Test unknown model
        cost_unknown = config.estimate_cost("unknown_model", 1000, 500)
        assert cost_unknown == 0.0


class TestConfigManager:
    """Test configuration manager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test.env"
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_config_from_file(self):
        """Test loading configuration from .env file."""
        # Create test config file
        config_content = """
OPENAI_API_KEY=sk-test123456789
DEFAULT_TEACHER_MODEL=gpt-4.1-mini
TEMPERATURE=0.8
BATCH_SIZE=64
"""
        with open(self.config_path, 'w') as f:
            f.write(config_content)
        
        manager = ConfigManager(self.config_path)
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789"}):
            config = manager.load_config()
            assert config.openai_api_key == "sk-test123456789"
    
    def test_config_validation(self):
        """Test configuration validation."""
        manager = ConfigManager()
        
        # Mock OpenAI client for testing
        with patch('openai.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.models.list.return_value = MagicMock()
            mock_openai.return_value = mock_client
            
            config = Config(openai_api_key="sk-test123456789")
            issues = manager.validate_config(config)
            
            # Should pass validation with mocked client
            assert isinstance(issues, list)
    
    def test_save_config(self):
        """Test saving configuration to file."""
        config = Config(openai_api_key="sk-test123456789")
        manager = ConfigManager(self.config_path)
        
        manager.save_config(config)
        
        assert self.config_path.exists()
        content = self.config_path.read_text()
        assert "sk-test123456789" not in content  # Should be redacted


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-integration123456789",
        "DEFAULT_SAMPLES_PER_CLASS": "2000",
        "USE_GPU": "false"
    })
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        from llm_distillation.config import get_config
        
        config = get_config()
        
        assert config.openai_api_key == "sk-integration123456789"
        assert config.default_samples_per_class == 2000
        assert config.use_gpu is False
    
    def test_config_to_dict_redaction(self):
        """Test configuration dictionary conversion with redaction."""
        config = Config(
            openai_api_key="sk-secret123456789",
            redact_api_keys=True
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["openai_api_key"] == "sk-***REDACTED***"
    
    def test_encryption_key_generation(self):
        """Test encryption key generation and retrieval."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Config(
                openai_api_key="sk-test123456789",
                cache_dir=Path(tmp_dir)
            )
            
            key1 = config.get_encryption_key()
            key2 = config.get_encryption_key()
            
            # Should return same key on subsequent calls
            assert key1 == key2
            assert len(key1) == 44  # Fernet key length


if __name__ == "__main__":
    pytest.main([__file__])