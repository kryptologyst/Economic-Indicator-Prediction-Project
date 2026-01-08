"""
Core utilities for the Economic Indicator Prediction project.

This module provides common utilities including seeding, device management,
logging, and configuration handling.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: Logging level.
        log_file: Optional log file path.
        format_string: Custom format string.
        
    Returns:
        Configured logger.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else []),
        ],
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        OmegaConf configuration object.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save.
        output_path: Output file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        OmegaConf.save(config, f)


def create_project_dirs(base_path: Union[str, Path]) -> None:
    """Create standard project directory structure.
    
    Args:
        base_path: Base project path.
    """
    base_path = Path(base_path)
    dirs = [
        "data/raw",
        "data/processed",
        "data/external",
        "assets/plots",
        "assets/models",
        "assets/results",
        "logs",
        "configs",
        "scripts",
        "notebooks",
        "tests",
        "demo",
        "docs",
    ]
    
    for dir_path in dirs:
        (base_path / dir_path).mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """Configuration manager for the project."""
    
    def __init__(self, config_path: Union[str, Path]):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config = load_config(config_path)
        self.base_config = self.config.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.
            
        Returns:
            Configuration value.
        """
        return OmegaConf.select(self.config, key, default=default)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of updates.
        """
        self.config = OmegaConf.merge(self.config, updates)
    
    def reset(self) -> None:
        """Reset configuration to base values."""
        self.config = self.base_config.copy()
    
    def save(self, output_path: Union[str, Path]) -> None:
        """Save current configuration.
        
        Args:
            output_path: Output file path.
        """
        save_config(self.config, output_path)


def validate_config(config: DictConfig) -> bool:
    """Validate configuration structure.
    
    Args:
        config: Configuration to validate.
        
    Returns:
        True if valid, False otherwise.
    """
    required_sections = ["data", "model", "training", "evaluation"]
    
    for section in required_sections:
        if section not in config:
            logging.error(f"Missing required configuration section: {section}")
            return False
    
    return True
