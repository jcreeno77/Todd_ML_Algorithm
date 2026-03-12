import pytest
import torch
import numpy as np


@pytest.fixture
def model_config():
    """Standard TFT hyperparameters for testing."""
    return {
        "hidden_size": 160,
        "lstm_layers": 2,
        "attention_heads": 4,
        "dropout": 0.3,
        "num_temporal_features": 47,
        "num_static_continuous": 7,
        "num_static_categorical": 1,
        "categorical_cardinalities": [11],  # 11 GICS sectors
        "categorical_embedding_dim": 8,
        "sequence_length": 30,
    }


@pytest.fixture
def sample_batch(model_config):
    """A batch of 4 samples with correct shapes."""
    batch_size = 4
    return {
        "temporal": torch.randn(batch_size, model_config["sequence_length"], model_config["num_temporal_features"]),
        "static_continuous": torch.randn(batch_size, model_config["num_static_continuous"]),
        "static_categorical": torch.randint(0, 11, (batch_size, model_config["num_static_categorical"])),
        "y_win": torch.randint(0, 2, (batch_size, 1)).float(),
        "y_offset": torch.randn(batch_size, 1),
    }
