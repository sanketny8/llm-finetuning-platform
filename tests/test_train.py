"""Tests for train.py -- config loading, argument parsing, and function signatures."""

import argparse
import os
import tempfile

import pytest
import yaml

# We import only the pure-Python helpers from train.py.
# The heavy ML imports (torch, transformers, etc.) happen at module level,
# so we must make them available or mock them before importing train.
# For CI environments without GPU libs, we patch at the top.

import importlib
import sys
from unittest.mock import MagicMock

# Provide lightweight stubs for heavy GPU-only packages so the test suite
# can run without installing torch, transformers, etc.
_STUB_MODULES = [
    "torch",
    "datasets",
    "transformers",
    "peft",
    "trl",
    "mlflow",
]

for mod_name in _STUB_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Submodules referenced in train.py's import block
for sub in [
    "transformers.AutoModelForCausalLM",
    "transformers.AutoTokenizer",
    "transformers.BitsAndBytesConfig",
    "transformers.TrainingArguments",
    "peft.LoraConfig",
    "peft.get_peft_model",
    "peft.prepare_model_for_kbit_training",
    "trl.SFTTrainer",
]:
    if sub not in sys.modules:
        sys.modules[sub] = MagicMock()

import train  # noqa: E402  (must come after stubs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config_path(tmp_path):
    """Write a minimal YAML config and return its path."""
    config = {
        "model": {"name": "test-model/small", "load_in_4bit": True},
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.1,
        },
        "training": {
            "output_dir": str(tmp_path / "output"),
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "learning_rate": 1e-4,
            "max_seq_length": 512,
        },
        "optimization": {
            "optim": "adamw_torch",
            "lr_scheduler_type": "linear",
        },
        "dataset": {
            "train_file": "data/train.json",
            "text_field": "content",
        },
    }
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml.dump(config))
    return str(config_file)


# ---------------------------------------------------------------------------
# Tests: load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_load_valid_yaml(self, sample_config_path):
        config = train.load_config(sample_config_path)
        assert isinstance(config, dict)
        assert config["model"]["name"] == "test-model/small"
        assert config["lora"]["r"] == 8

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            train.load_config("/nonexistent/path/config.yaml")

    def test_load_empty_yaml(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        result = train.load_config(str(empty))
        assert result is None


# ---------------------------------------------------------------------------
# Tests: parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_cli_flags_only(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            [
                "train.py",
                "--model_name", "meta-llama/Llama-3-8B",
                "--dataset", "data/train.json",
                "--output_dir", "./out",
                "--lora_r", "8",
                "--epochs", "1",
            ],
        )
        args = train.parse_args()
        assert args.model_name == "meta-llama/Llama-3-8B"
        assert args.dataset == "data/train.json"
        assert args.lora_r == 8
        assert args.epochs == 1

    def test_config_file_only(self, monkeypatch, sample_config_path):
        monkeypatch.setattr(
            "sys.argv",
            ["train.py", "--config", sample_config_path],
        )
        args = train.parse_args()
        assert args.model_name == "test-model/small"
        assert args.dataset == "data/train.json"
        assert args.use_qlora is True
        assert args.lora_r == 8
        assert args.lora_alpha == 16
        assert args.lora_dropout == 0.1
        assert args.learning_rate == 1e-4
        assert args.text_field == "content"

    def test_cli_overrides_config(self, monkeypatch, sample_config_path):
        monkeypatch.setattr(
            "sys.argv",
            [
                "train.py",
                "--config", sample_config_path,
                "--learning_rate", "5e-5",
                "--epochs", "10",
            ],
        )
        args = train.parse_args()
        # CLI values should win
        assert args.learning_rate == 5e-5
        assert args.epochs == 10
        # Config values still used for others
        assert args.model_name == "test-model/small"

    def test_missing_required_fields(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["train.py"])
        with pytest.raises(ValueError, match="--model_name is required"):
            train.parse_args()

    def test_missing_dataset(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["train.py", "--model_name", "some-model"],
        )
        with pytest.raises(ValueError, match="--dataset is required"):
            train.parse_args()


# ---------------------------------------------------------------------------
# Tests: function signatures
# ---------------------------------------------------------------------------

class TestFunctionSignatures:
    def test_train_is_callable(self):
        assert callable(train.train)

    def test_main_is_callable(self):
        assert callable(train.main)

    def test_load_config_is_callable(self):
        assert callable(train.load_config)

    def test_parse_args_is_callable(self):
        assert callable(train.parse_args)
