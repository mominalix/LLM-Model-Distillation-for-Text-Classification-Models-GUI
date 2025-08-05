"""
Pytest configuration and shared fixtures.

This module provides common test fixtures and configuration
for the entire test suite.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from llm_distillation.config import Config


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_config(temp_dir):
    """Provide a test configuration."""
    return Config(
        openai_api_key="sk-test123456789abcdef",
        datasets_dir=temp_dir / "datasets",
        models_dir=temp_dir / "models",
        cache_dir=temp_dir / "cache",
        logs_dir=temp_dir / "logs",
        output_dir=temp_dir / "output",
        enable_audit_logging=True,
        use_gpu=False,  # Disable GPU for tests
        default_samples_per_class=10,  # Small number for fast tests
        num_epochs=1,  # Quick training for tests
        batch_size=2   # Small batches for tests
    )


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch('openai.OpenAI') as mock_client_class:
        mock_client = MagicMock()
        
        # Mock chat completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"examples": [{"text": "test", "label": "positive"}]}'
        mock_response.choices[0].finish_reason = "completed"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_response.id = "test-response-id"
        mock_response.created = 1234567890
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Mock models list
        mock_models = MagicMock()
        mock_models.data = [MagicMock(id="gpt-4.1-nano")]
        mock_client.models.list.return_value = mock_models
        
        mock_client_class.return_value = mock_client
        
        yield mock_client


@pytest.fixture
def mock_huggingface_model():
    """Mock Hugging Face model for testing."""
    with patch('transformers.AutoModelForSequenceClassification') as mock_model_class:
        mock_model = MagicMock()
        mock_model.num_labels = 3
        mock_model.config.to_dict.return_value = {"test": "config"}
        
        # Mock forward pass
        mock_output = MagicMock()
        mock_output.logits = MagicMock()
        mock_model.return_value = mock_output
        
        mock_model_class.from_pretrained.return_value = mock_model
        
        yield mock_model


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    with patch('transformers.AutoTokenizer') as mock_tokenizer_class:
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        
        # Mock tokenization output
        mock_tokenizer.return_value = {
            'input_ids': [[1, 2, 3, 4, 5]],
            'attention_mask': [[1, 1, 1, 1, 1]]
        }
        
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        yield mock_tokenizer


@pytest.fixture
def sample_dataset():
    """Provide sample dataset for testing."""
    return [
        {"text": "This is a positive example", "label": "positive"},
        {"text": "This is a negative example", "label": "negative"},
        {"text": "This is neutral text", "label": "neutral"},
        {"text": "Another positive case", "label": "positive"},
        {"text": "Another negative case", "label": "negative"},
        {"text": "More neutral content", "label": "neutral"},
    ]


@pytest.fixture
def sample_generation_task():
    """Provide sample generation task for testing."""
    from llm_distillation.data import GenerationTask
    from llm_distillation.config import Language
    
    return GenerationTask(
        task_description="Classify text sentiment",
        class_labels=["positive", "negative", "neutral"],
        samples_per_class=5,  # Small number for fast tests
        language=Language.EN,
        temperature=0.7,
        quality_threshold=0.6
    )


@pytest.fixture(autouse=True)
def disable_gpu():
    """Disable GPU for all tests to ensure consistent behavior."""
    with patch('torch.cuda.is_available', return_value=False):
        yield


@pytest.fixture(autouse=True)
def mock_nltk_downloads():
    """Mock NLTK downloads to avoid network calls in tests."""
    with patch('nltk.download'):
        yield


@pytest.fixture
def mock_environment():
    """Mock environment variables for testing."""
    test_env = {
        'OPENAI_API_KEY': 'sk-test123456789abcdef',
        'DEFAULT_TEACHER_MODEL': 'gpt-4.1-nano',
        'DEFAULT_STUDENT_MODEL': 'distilbert-base-uncased',
        'USE_GPU': 'false',
        'LOG_LEVEL': 'DEBUG'
    }
    
    with patch.dict(os.environ, test_env, clear=False):
        yield test_env


@pytest.fixture
def disable_logging():
    """Disable logging during tests to reduce noise."""
    import logging
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


# Test marks for categorizing tests
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.gpu = pytest.mark.gpu
pytest.mark.api = pytest.mark.api


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "api: mark test as requiring API access")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit test marker to all tests by default
        if not any(marker.name in ['integration', 'slow', 'gpu', 'api'] 
                  for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker to tests that likely take longer
        if 'test_training' in item.name or 'test_generation' in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Add API marker to tests that use external APIs
        if 'openai' in item.name.lower() or 'api' in item.name.lower():
            item.add_marker(pytest.mark.api)