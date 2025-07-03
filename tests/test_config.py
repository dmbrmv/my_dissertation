"""Tests for configuration settings."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config.settings import Settings, DataConfig, ModelConfig, TrainingConfig


class TestDataConfig:
    """Test data configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()
        
        assert config.meteo_input == ["prcp_e5l", "t_max_e5l", "t_min_e5l"]
        assert config.hydro_target == "q_mm_day"
        assert config.index_col == "gauge_id"
    
    def test_hydro_target_validation(self):
        """Test hydrological target validation."""
        # Valid targets should pass
        for target in ["q_mm_day", "lvl_sm", "lvl_mbs"]:
            config = DataConfig(hydro_target=target)
            assert config.hydro_target == target
        
        # Invalid target should raise error
        with pytest.raises(ValueError, match="hydro_target must be one of"):
            DataConfig(hydro_target="invalid_target")


class TestModelConfig:
    """Test model configuration."""
    
    def test_default_values(self):
        """Test default model configuration."""
        config = ModelConfig()
        
        assert config.input_chunk_length == 90
        assert config.output_chunk_length == 30
        assert config.hidden_size == 128
        assert config.batch_size == 32
        assert config.likelihood == "quantile"
        assert 0.5 in config.quantiles
    
    def test_quantiles_validation(self):
        """Test quantiles validation."""
        # Valid quantiles should pass
        valid_quantiles = [0.1, 0.5, 0.9]
        config = ModelConfig(quantiles=valid_quantiles)
        assert config.quantiles == [0.1, 0.5, 0.9]
        
        # Missing median should raise error
        with pytest.raises(ValueError, match="Quantiles must include 0.5"):
            ModelConfig(quantiles=[0.1, 0.9])
        
        # Invalid quantile values should raise error
        with pytest.raises(ValueError, match="All quantiles must be between 0 and 1"):
            ModelConfig(quantiles=[0.1, 0.5, 1.5])


class TestTrainingConfig:
    """Test training configuration."""
    
    def test_default_values(self):
        """Test default training configuration."""
        config = TrainingConfig()
        
        assert config.train_split == 0.7
        assert config.val_split == 0.2
        assert config.test_split == 0.1
        assert config.early_stopping_patience == 10
    
    def test_splits_sum_to_one(self):
        """Test that data splits sum to 1.0."""
        # Valid splits should pass
        config = TrainingConfig(
            train_split=0.6,
            val_split=0.3,
            test_split=0.1
        )
        assert abs(config.train_split + config.val_split + config.test_split - 1.0) < 1e-6
        
        # Invalid splits should raise error
        with pytest.raises(ValueError, match="must equal 1.0"):
            TrainingConfig(
                train_split=0.6,
                val_split=0.3,
                test_split=0.2  # Sum > 1.0
            )


class TestSettings:
    """Test main settings class."""
    
    def test_default_initialization(self):
        """Test default settings initialization."""
        settings = Settings()
        
        assert isinstance(settings.data, DataConfig)
        assert isinstance(settings.model, ModelConfig)
        assert isinstance(settings.training, TrainingConfig)
    
    def test_update_static_parameters(self):
        """Test static parameters update based on target."""
        settings = Settings()
        
        # Test for discharge target
        settings.data.hydro_target = "q_mm_day"
        settings.update_static_parameters()
        
        assert "height_bs" not in settings.data.static_parameters
        assert settings.data.nc_variable == "nc_all_q"
        
        # Test for level target
        settings.data.hydro_target = "lvl_sm"
        settings.update_static_parameters()
        
        assert "height_bs" in settings.data.static_parameters
        assert settings.data.nc_variable == "nc_all_h"
    
    def test_yaml_roundtrip(self):
        """Test saving and loading settings to/from YAML."""
        settings = Settings()
        settings.model.hidden_size = 256
        settings.training.early_stopping_patience = 15
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Save to YAML
            settings.to_yaml(temp_path)
            
            # Load from YAML
            loaded_settings = Settings.from_yaml(temp_path)
            
            # Check that values are preserved
            assert loaded_settings.model.hidden_size == 256
            assert loaded_settings.training.early_stopping_patience == 15
            
        finally:
            temp_path.unlink()
    
    def test_from_yaml_with_partial_config(self):
        """Test loading settings from partial YAML configuration."""
        partial_config = {
            "model": {
                "hidden_size": 64,
                "n_epochs": 50
            },
            "training": {
                "train_split": 0.8
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(partial_config, f)
            temp_path = Path(f.name)
        
        try:
            settings = Settings.from_yaml(temp_path)
            
            # Check that specified values are loaded
            assert settings.model.hidden_size == 64
            assert settings.model.n_epochs == 50
            assert settings.training.train_split == 0.8
            
            # Check that default values are preserved
            assert settings.model.dropout == 0.1  # Default value
            assert settings.training.val_split == 0.2  # Default value
            
        finally:
            temp_path.unlink()
