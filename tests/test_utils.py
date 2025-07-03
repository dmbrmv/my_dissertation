"""Tests for utility functions."""

import tempfile
from pathlib import Path

import torch
import pytest

from src.utils.helpers import (
    set_random_seed,
    check_device,
    ensure_directory,
    format_duration,
    create_experiment_name,
    validate_file_exists
)


class TestRandomSeed:
    """Test random seed utilities."""
    
    def test_set_random_seed(self):
        """Test setting random seed for reproducibility."""
        set_random_seed(42)
        
        # Generate some random numbers
        torch_val1 = torch.randn(1).item()
        
        # Reset seed and generate again
        set_random_seed(42)
        torch_val2 = torch.randn(1).item()
        
        # Should be identical
        assert abs(torch_val1 - torch_val2) < 1e-6


class TestDeviceCheck:
    """Test device checking utilities."""
    
    def test_check_device(self):
        """Test device checking returns valid device."""
        device = check_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda"]


class TestDirectoryUtils:
    """Test directory utilities."""
    
    def test_ensure_directory(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "new_dir" / "nested_dir"
            
            # Directory shouldn't exist initially
            assert not test_path.exists()
            
            # Create directory
            result = ensure_directory(test_path)
            
            # Should exist now and return the path
            assert test_path.exists()
            assert test_path.is_dir()
            assert result == test_path
    
    def test_ensure_existing_directory(self):
        """Test ensuring an already existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir)
            
            # Should not raise error
            result = ensure_directory(test_path)
            assert result == test_path


class TestFormatting:
    """Test formatting utilities."""
    
    def test_format_duration(self):
        """Test duration formatting."""
        # Test seconds only
        assert format_duration(30.5) == "30.5s"
        
        # Test minutes and seconds
        assert format_duration(90.0) == "1m 30.0s"
        
        # Test hours, minutes, and seconds
        assert format_duration(3661.5) == "1h 1m 1.5s"
        
        # Test zero duration
        assert format_duration(0.0) == "0.0s"
    
    def test_create_experiment_name(self):
        """Test experiment name creation."""
        # Basic name
        name = create_experiment_name("test", include_timestamp=False)
        assert name == "test"
        
        # With extra tags
        name = create_experiment_name(
            "test", 
            include_timestamp=False, 
            extra_tags=["single", "gauge"]
        )
        assert name == "test_single_gauge"
        
        # With timestamp (should contain timestamp pattern)
        name = create_experiment_name("test", include_timestamp=True)
        assert name.startswith("test_")
        assert len(name.split("_")[-1]) == 15  # YYYYMMDD_HHMMSS format


class TestFileValidation:
    """Test file validation utilities."""
    
    def test_validate_existing_file(self):
        """Test validation of existing file."""
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_path = Path(temp_file.name)
            
            # Should not raise error
            validate_file_exists(temp_path, "Test file")
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        nonexistent_path = Path("/this/file/does/not/exist.txt")
        
        with pytest.raises(FileNotFoundError, match="Test file not found"):
            validate_file_exists(nonexistent_path, "Test file")
    
    def test_validate_directory_as_file(self):
        """Test validation when path points to directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            with pytest.raises(ValueError, match="is not a file"):
                validate_file_exists(temp_path, "Test file")
