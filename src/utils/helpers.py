"""Utility functions and helpers."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np
import random


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional path to log file
        format_string: Optional custom format string
        
    Examples:
        >>> setup_logging(level="DEBUG", log_file=Path("training.log"))
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure logging
    logging_config = {
        "level": getattr(logging, level.upper()),
        "format": format_string,
        "datefmt": "%Y-%m-%d %H:%M:%S"
    }
    
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logging_config["filename"] = str(log_file)
        logging_config["filemode"] = "a"
    
    logging.basicConfig(**logging_config)
    
    # Reduce noise from external libraries
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("darts").setLevel(logging.WARNING)


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
        
    Examples:
        >>> set_random_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_device() -> torch.device:
    """Check and return the best available device.
    
    Returns:
        PyTorch device object
        
    Examples:
        >>> device = check_device()
        >>> print(f"Using device: {device}")
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure
        
    Returns:
        The directory path
        
    Examples:
        >>> output_dir = ensure_directory(Path("outputs/models"))
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dict_to_yaml(data: Dict[str, Any], path: Path) -> None:
    """Save dictionary to YAML file.
    
    Args:
        data: Dictionary to save
        path: Output file path
        
    Examples:
        >>> config = {"model": {"lr": 0.001}}
        >>> save_dict_to_yaml(config, Path("config.yaml"))
    """
    import yaml
    
    ensure_directory(path.parent)
    
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_yaml_to_dict(path: Path) -> Dict[str, Any]:
    """Load YAML file to dictionary.
    
    Args:
        path: YAML file path
        
    Returns:
        Dictionary with loaded data
        
    Examples:
        >>> config = load_yaml_to_dict(Path("config.yaml"))
    """
    import yaml
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
        
    Examples:
        >>> duration_str = format_duration(3661.5)
        >>> print(duration_str)  # "1h 1m 1.5s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs:.1f}s")
    
    return " ".join(parts)


def print_memory_usage() -> None:
    """Print current GPU memory usage if CUDA is available.
    
    Examples:
        >>> print_memory_usage()
        GPU Memory: 2.1 GB / 8.0 GB (26.3%)
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        total = torch.cuda.get_device_properties(0).total_memory
        
        allocated_gb = allocated / 1024**3
        total_gb = total / 1024**3
        percentage = (allocated / total) * 100
        
        print(f"GPU Memory: {allocated_gb:.1f} GB / {total_gb:.1f} GB ({percentage:.1f}%)")
    else:
        print("GPU not available")


def create_experiment_name(
    base_name: str = "tft_experiment",
    include_timestamp: bool = True,
    extra_tags: Optional[List[str]] = None
) -> str:
    """Create a unique experiment name.
    
    Args:
        base_name: Base name for the experiment
        include_timestamp: Whether to include timestamp
        extra_tags: Additional tags to include
        
    Returns:
        Formatted experiment name
        
    Examples:
        >>> name = create_experiment_name("tft", extra_tags=["single_gauge"])
        >>> print(name)  # "tft_single_gauge_20240703_142030"
    """
    from datetime import datetime
    
    parts = [base_name]
    
    if extra_tags:
        parts.extend(extra_tags)
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(timestamp)
    
    return "_".join(parts)


def validate_file_exists(path: Path, description: str = "File") -> None:
    """Validate that a file exists and raise informative error if not.
    
    Args:
        path: Path to validate
        description: Description of the file for error message
        
    Raises:
        FileNotFoundError: If file doesn't exist
        
    Examples:
        >>> validate_file_exists(Path("data.csv"), "Input data file")
    """
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    
    if not path.is_file():
        raise ValueError(f"{description} is not a file: {path}")


def print_model_summary(
    model_path: Path,
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """Print a summary of the trained model.
    
    Args:
        model_path: Path to the saved model
        metrics: Optional metrics to display
        
    Examples:
        >>> metrics = {"NSE": 0.75, "KGE": 0.68}
        >>> print_model_summary(Path("model.pkl"), metrics)
    """
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    print(f"Model saved at: {model_path}")
    print(f"File size: {model_path.stat().st_size / 1024**2:.1f} MB")
    
    if metrics:
        print("\nPerformance Metrics:")
        print("-" * 20)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    print("="*50)
