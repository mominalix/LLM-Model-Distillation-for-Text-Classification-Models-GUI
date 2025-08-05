"""
Integration tests for the LLM Distillation application.

This module tests the integration between different components
to ensure the complete workflow functions correctly.
"""

import pytest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

from llm_distillation.config import Config
from llm_distillation.llm import ModelManager
from llm_distillation.data import DataGenerator, GenerationTask
from llm_distillation.training import ModelTrainer, TrainingConfig
from llm_distillation.security import SecurityManager


@pytest.mark.integration
class TestDataGenerationWorkflow:
    """Test the complete data generation workflow."""
    
    @pytest.fixture
    def setup_components(self, test_config, mock_openai_client):
        """Set up components for data generation testing."""
        model_manager = ModelManager(test_config)
        data_generator = DataGenerator(test_config, model_manager)
        security_manager = SecurityManager(test_config)
        
        return {
            'config': test_config,
            'model_manager': model_manager,
            'data_generator': data_generator,
            'security_manager': security_manager
        }
    
    def test_end_to_end_data_generation(self, setup_components, sample_generation_task):
        """Test complete data generation from task to dataset."""
        components = setup_components
        data_generator = components['data_generator']
        
        # Mock the LLM response to return valid JSON
        mock_response = [
            {"text": "This is great!", "label": "positive", "confidence": 0.9},
            {"text": "This is terrible!", "label": "negative", "confidence": 0.9},
            {"text": "This is okay.", "label": "neutral", "confidence": 0.8}
        ]
        
        with patch.object(data_generator, '_parse_generation_response', return_value=mock_response):
            result = data_generator.generate_dataset(
                sample_generation_task,
                model_name="gpt-4.1-nano"
            )
            
            assert result is not None
            assert len(result.generated_samples) > 0
            assert result.success_rate > 0
            assert result.quality_metrics is not None
    
    def test_data_generation_with_security_scanning(self, setup_components, sample_generation_task):
        """Test data generation with PII scanning integration."""
        components = setup_components
        data_generator = components['data_generator']
        security_manager = components['security_manager']
        
        # Mock generation response with PII
        mock_response = [
            {"text": "Contact me at john@example.com", "label": "positive", "confidence": 0.9},
            {"text": "This is safe text", "label": "negative", "confidence": 0.9}
        ]
        
        with patch.object(data_generator, '_parse_generation_response', return_value=mock_response):
            result = data_generator.generate_dataset(sample_generation_task)
            
            # Scan generated samples for PII
            pii_found = False
            for sample in result.generated_samples:
                pii_result, is_safe = security_manager.scan_text_for_pii(
                    sample['text'],
                    context="training_data"
                )
                if pii_result.has_pii:
                    pii_found = True
            
            assert result is not None
            # We expect PII to be found in the mocked response
            assert pii_found


@pytest.mark.integration
class TestTrainingWorkflow:
    """Test the complete model training workflow."""
    
    @pytest.fixture
    def setup_training_components(self, test_config, mock_huggingface_model, mock_tokenizer):
        """Set up components for training testing."""
        model_manager = ModelManager(test_config)
        
        training_config = TrainingConfig(
            task_name="test_classification",
            model_name="distilbert-base-uncased",
            output_dir=str(test_config.output_dir / "training"),
            num_epochs=1,
            batch_size=2,
            use_distillation=False  # Disable for simpler testing
        )
        
        trainer = ModelTrainer(test_config, training_config, model_manager)
        
        return {
            'config': test_config,
            'training_config': training_config,
            'trainer': trainer,
            'model_manager': model_manager
        }
    
    def test_data_preparation_for_training(self, setup_training_components, sample_dataset):
        """Test data preparation pipeline for training."""
        components = setup_training_components
        trainer = components['trainer']
        
        # Prepare data
        dataset = trainer.prepare_data(sample_dataset, validate_data=False)
        
        assert dataset is not None
        assert 'train' in dataset
        assert len(dataset['train']) > 0
        
        # Check dataset structure
        train_dataset = dataset['train']
        assert hasattr(train_dataset, 'label_to_id')
        assert hasattr(train_dataset, 'id_to_label')
        assert hasattr(train_dataset, 'num_labels')
    
    @patch('transformers.Trainer')
    def test_training_pipeline(self, mock_trainer_class, setup_training_components, sample_dataset):
        """Test the training pipeline with mocked trainer."""
        components = setup_training_components
        trainer = components['trainer']
        
        # Mock the Trainer
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = MagicMock(training_loss=0.5)
        mock_trainer.evaluate.return_value = {'eval_accuracy': 0.8, 'eval_f1': 0.75}
        mock_trainer_class.return_value = mock_trainer
        
        # Prepare and train
        dataset = trainer.prepare_data(sample_dataset, validate_data=False)
        
        # Mock the training process
        with patch.object(trainer, 'prepare_data', return_value=dataset):
            result = trainer.train()
            
            assert result is not None
            assert 'training_type' in result
            assert result['training_type'] == 'standard'


@pytest.mark.integration
class TestSecurityIntegration:
    """Test security integration across components."""
    
    def test_security_manager_with_all_components(self, test_config):
        """Test security manager integration with all components."""
        # Initialize all components
        model_manager = ModelManager(test_config)
        security_manager = SecurityManager(test_config)
        
        # Test API access validation
        api_allowed, reason = security_manager.validate_api_access(
            "openai",
            user_id="test_user"
        )
        assert api_allowed
        
        # Test PII scanning
        pii_result, is_safe = security_manager.scan_text_for_pii(
            "Contact support at help@company.com",
            context="user_input",
            user_id="test_user"
        )
        assert pii_result.has_pii
        
        # Test data export validation
        export_allowed, reason = security_manager.validate_data_export(
            data_size_mb=10.0,
            export_type="dataset",
            user_id="test_user"
        )
        assert export_allowed
        
        # Get security status
        status = security_manager.get_security_status()
        assert status['performance_metrics']['pii_scans'] > 0
    
    def test_audit_logging_across_components(self, test_config):
        """Test that audit logging works across different components."""
        security_manager = SecurityManager(test_config)
        audit_logger = security_manager.audit_logger
        
        # Simulate various operations
        audit_logger.log_user_action("button_click", {"button": "generate"})
        audit_logger.log_api_call("openai", "POST", 200, 1500.0, 0.01)
        audit_logger.log_data_generation("started", 0)
        
        # Check statistics
        stats = audit_logger.get_event_statistics()
        assert stats['total_events'] >= 3
        
        # Check recent events
        recent_events = audit_logger.get_recent_events(10)
        assert len(recent_events) >= 3


@pytest.mark.integration
@pytest.mark.slow
class TestFullApplicationWorkflow:
    """Test the complete application workflow end-to-end."""
    
    def test_complete_workflow_simulation(self, test_config, mock_openai_client, mock_huggingface_model, mock_tokenizer):
        """Simulate a complete user workflow through the application."""
        
        # Initialize main components
        model_manager = ModelManager(test_config)
        security_manager = SecurityManager(test_config)
        data_generator = DataGenerator(test_config, model_manager)
        
        # Step 1: Configure generation task
        from llm_distillation.data import GenerationTask
        from llm_distillation.config import Language
        
        task = GenerationTask(
            task_description="Classify customer feedback sentiment",
            class_labels=["positive", "negative", "neutral"],
            samples_per_class=3,
            language=Language.EN
        )
        
        # Step 2: Generate synthetic data
        mock_samples = [
            {"text": "Great service!", "label": "positive", "confidence": 0.9},
            {"text": "Poor experience", "label": "negative", "confidence": 0.9},
            {"text": "It's okay", "label": "neutral", "confidence": 0.8}
        ] * 3  # 9 total samples
        
        with patch.object(data_generator, '_parse_generation_response', return_value=mock_samples):
            generation_result = data_generator.generate_dataset(task)
            
            assert generation_result is not None
            assert len(generation_result.generated_samples) > 0
        
        # Step 3: Security scan the generated data
        safe_samples = []
        for sample in generation_result.generated_samples:
            pii_result, is_safe = security_manager.scan_text_for_pii(
                sample['text'],
                context="training_data"
            )
            if is_safe:
                safe_samples.append(sample)
        
        assert len(safe_samples) > 0
        
        # Step 4: Prepare for training
        training_config = TrainingConfig(
            task_name="sentiment_classification",
            model_name="distilbert-base-uncased",
            output_dir=str(test_config.output_dir / "training"),
            num_epochs=1,
            batch_size=2,
            use_distillation=False
        )
        
        trainer = ModelTrainer(test_config, training_config, model_manager)
        
        # Prepare data
        dataset = trainer.prepare_data(safe_samples, validate_data=False)
        assert dataset is not None
        
        # Step 5: Validate export permissions
        export_allowed, reason = security_manager.validate_data_export(
            data_size_mb=1.0,
            export_type="dataset"
        )
        assert export_allowed
        
        # Step 6: Check final security status
        security_status = security_manager.get_security_status()
        assert security_status['performance_metrics']['pii_scans'] > 0
        
        # Step 7: Verify audit trail
        if security_manager.audit_logger:
            audit_stats = security_manager.audit_logger.get_event_statistics()
            assert audit_stats['total_events'] > 0


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""
    
    def test_generation_failure_handling(self, test_config, sample_generation_task):
        """Test handling of generation failures."""
        model_manager = ModelManager(test_config)
        data_generator = DataGenerator(test_config, model_manager)
        
        # Mock a failure in the OpenAI client
        with patch.object(model_manager, 'get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.generate_text.side_effect = Exception("API Error")
            mock_get_model.return_value = mock_model
            
            # Should handle the error gracefully
            with pytest.raises(Exception):
                data_generator.generate_dataset(sample_generation_task)
    
    def test_security_scanning_failure_recovery(self, test_config):
        """Test recovery from security scanning failures."""
        # Create security manager with intentionally broken PII detector
        security_manager = SecurityManager(test_config)
        
        # Mock PII detector to fail
        with patch.object(security_manager.pii_detector, 'detect_pii') as mock_detect:
            mock_detect.side_effect = Exception("PII Detection Error")
            
            # Should fail safe (assume unsafe)
            pii_result, is_safe = security_manager.scan_text_for_pii("test text")
            
            assert not is_safe  # Should fail safe
            assert security_manager.performance_metrics['security_violations'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])