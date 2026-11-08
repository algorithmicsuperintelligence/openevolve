"""
Test CLI model override functionality (GitHub issue #245)
"""
import asyncio
import sys
import unittest
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch

from openevolve.cli import main_async
from openevolve.config import Config, load_config


class TestCLIModelOverride(unittest.TestCase):
    """Test that CLI model overrides work correctly"""
    
    def test_rebuild_models_with_both_models(self):
        """Test rebuilding models with both primary and secondary models"""
        config = Config()
        
        # Initially no models
        self.assertEqual(len(config.llm.models), 0)
        
        # Set CLI overrides
        config.llm.primary_model = "gpt-4"
        config.llm.secondary_model = "gpt-3.5-turbo"
        
        # Models list should still be empty before rebuild
        self.assertEqual(len(config.llm.models), 0)
        
        # Rebuild models
        config.llm.rebuild_models()
        
        # Now should have both models
        self.assertEqual(len(config.llm.models), 2)
        self.assertEqual(config.llm.models[0].name, "gpt-4")
        self.assertEqual(config.llm.models[0].weight, 1.0)
        self.assertEqual(config.llm.models[1].name, "gpt-3.5-turbo")
        self.assertEqual(config.llm.models[1].weight, 0.2)
    
    def test_rebuild_models_primary_only(self):
        """Test rebuilding with only primary model"""
        config = Config()
        config.llm.primary_model = "claude-3-opus"
        
        config.llm.rebuild_models()
        
        self.assertEqual(len(config.llm.models), 1)
        self.assertEqual(config.llm.models[0].name, "claude-3-opus")
        self.assertEqual(config.llm.models[0].weight, 1.0)
    
    def test_rebuild_models_with_weights(self):
        """Test rebuilding with custom weights"""
        config = Config()
        config.llm.primary_model = "gpt-4"
        config.llm.primary_model_weight = 0.8
        config.llm.secondary_model = "gpt-3.5-turbo"
        config.llm.secondary_model_weight = 0.5
        
        config.llm.rebuild_models()
        
        self.assertEqual(len(config.llm.models), 2)
        self.assertEqual(config.llm.models[0].weight, 0.8)
        self.assertEqual(config.llm.models[1].weight, 0.5)
    
    def test_rebuild_models_zero_weight_secondary(self):
        """Test that secondary model with zero weight is excluded"""
        config = Config()
        config.llm.primary_model = "gpt-4"
        config.llm.secondary_model = "gpt-3.5-turbo"
        config.llm.secondary_model_weight = 0.0
        
        config.llm.rebuild_models()
        
        # Should only have primary model
        self.assertEqual(len(config.llm.models), 1)
        self.assertEqual(config.llm.models[0].name, "gpt-4")
    
    def test_rebuild_preserves_shared_config(self):
        """Test that rebuilding preserves shared configuration"""
        config = Config()
        config.llm.api_base = "https://custom-api.com/v1"
        config.llm.temperature = 0.8
        config.llm.primary_model = "custom-model"
        
        config.llm.rebuild_models()
        
        # Model should inherit shared configuration
        self.assertEqual(config.llm.models[0].api_base, "https://custom-api.com/v1")
        self.assertEqual(config.llm.models[0].temperature, 0.8)
    
    def test_rebuild_models_with_config_file_override(self):
        """Test CLI override of config file models"""
        config_content = """
llm:
  primary_model: "original-model"
  temperature: 0.5
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        try:
            # Load config from file
            config = load_config(config_path)
            
            # Verify original model is loaded
            self.assertEqual(config.llm.models[0].name, "original-model")
            
            # Apply CLI override
            config.llm.primary_model = "overridden-model"
            config.llm.rebuild_models()
            
            # Should now use overridden model
            self.assertEqual(len(config.llm.models), 1)
            self.assertEqual(config.llm.models[0].name, "overridden-model")
            # Should preserve other settings
            self.assertEqual(config.llm.temperature, 0.5)
            
        finally:
            os.unlink(config_path)
    
    def test_evaluator_models_updated_after_rebuild(self):
        """Test that evaluator_models list is also updated after rebuild"""
        config = Config()
        config.llm.primary_model = "test-model"
        
        config.llm.rebuild_models()
        
        # Evaluator models should be populated from main models
        self.assertEqual(len(config.llm.evaluator_models), 1)
        self.assertEqual(config.llm.evaluator_models[0].name, "test-model")

    def test_cli_api_base_alone_overrides_config_file_models(self):
        """--api-base without a model flag must reach the models from the config file"""
        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "config.yaml")
            with open(config_path, "w") as f:
                f.write('llm:\n  models:\n    - name: "gpt-4o-mini"\n      weight: 1.0\n')
            program_path = os.path.join(tmp, "program.py")
            with open(program_path, "w") as f:
                f.write("x = 1\n")
            eval_path = os.path.join(tmp, "evaluator.py")
            with open(eval_path, "w") as f:
                f.write("def evaluate(program_path):\n    return {'combined_score': 1.0}\n")

            argv = [
                "openevolve-run",
                program_path,
                eval_path,
                "--config",
                config_path,
                "--output",
                tmp,
                "--api-base",
                "http://localhost:8000/v1",
            ]
            with patch.object(sys, "argv", argv), patch("openevolve.cli.OpenEvolve") as mock_cls:
                mock_cls.return_value.run = AsyncMock(return_value=MagicMock(metrics={}))
                mock_cls.return_value.output_dir = tmp
                self.assertEqual(asyncio.run(main_async()), 0)

        config = mock_cls.call_args.kwargs["config"]
        for model in config.llm.models + config.llm.evaluator_models:
            self.assertEqual(model.api_base, "http://localhost:8000/v1")


if __name__ == '__main__':
    unittest.main()