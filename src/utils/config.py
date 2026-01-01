"""Configuration management utilities."""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set default paths
    config = _set_default_paths(config)
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def _set_default_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Set default paths in configuration."""
    # Ensure directories exist
    paths_to_create = [
        config.get('data', {}).get('data_dir', 'data/raw'),
        config.get('data', {}).get('processed_dir', 'data/processed'),
        config.get('training', {}).get('checkpoint_dir', 'checkpoints'),
        config.get('logging', {}).get('log_dir', 'logs'),
        config.get('reports', {}).get('output_dir', 'reports'),
    ]
    
    for path in paths_to_create:
        Path(path).mkdir(parents=True, exist_ok=True)
    
    return config


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the value (e.g., 'model.architecture')
        default: Default value if key not found
        
    Returns:
        Configuration value
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        config: Original configuration
        updates: Dictionary of updates
        
    Returns:
        Updated configuration
    """
    def _deep_update(base: Dict, updates: Dict) -> Dict:
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                _deep_update(base[key], value)
            else:
                base[key] = value
        return base
    
    return _deep_update(config.copy(), updates)
