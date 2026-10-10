import unittest
import os
import yaml
import tempfile
from pathlib import Path
from openevolve.config import render_config_dict, load_config

class TestConfigRender(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "test_config.yaml"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_legacy_rendering(self):
        config_content = """
base_val: 10
derived: "{{base_val}}"
nested:
  child: 5
  ref: "{{nested.child}}"
"""
        data = render_config_dict(config_content)
            
        self.assertEqual(data['derived'], 10)
        self.assertEqual(data['nested']['ref'], 5)

    def test_f_string_simple_expression(self):
        config_content = """
val: 100
expr: 'f"{val * 2}"'
"""
        data = render_config_dict(config_content)
            
        self.assertEqual(data['expr'], 200)

    def test_f_string_math_integration(self):
        config_content = """
val: 16
sqrt_val: 'f"{math.sqrt(val)}"'
"""
        data = render_config_dict(config_content)
            
        self.assertEqual(data['sqrt_val'], 4.0)

    def test_f_string_nested_context(self):
        config_content = """
database:
  num_islands: 4
  batch_size: 10
total_parallel: 'f"{database.num_islands * database.batch_size}"'
"""
        data = render_config_dict(config_content)
            
        self.assertEqual(data['total_parallel'], 40)

    def test_f_string_mixed_content(self):
        config_content = """
name: "Evolve"
msg: 'f"Hello {name}!"'
"""
        data = render_config_dict(config_content)
            
        self.assertEqual(data['msg'], "Hello Evolve!")

    def test_load_config_integration(self):
        config_content = """
llm:
  temperature: 'f"{0.5 + 0.2}"'
max_iterations: 'f"{10 * 100}"'
"""
        self.config_path.write_text(config_content)
        config = load_config(self.config_path)
        
        self.assertAlmostEqual(config.llm.temperature, 0.7)
        self.assertEqual(config.max_iterations, 1000)

if __name__ == "__main__":
    unittest.main()
