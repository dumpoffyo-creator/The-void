"""Configuration management for object detection system."""

import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Load and manage configuration from YAML files."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot notation key.
        
        Args:
            key: Configuration key (e.g., 'model.backbone')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
                if value is default:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot notation key.
        
        Args:
            key: Configuration key (e.g., 'model.backbone')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Path to save configuration (defaults to original path)
        """
        path = Path(output_path) if output_path else self.config_path
        
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def __repr__(self) -> str:
        """Return string representation of configuration."""
        return json.dumps(self.config, indent=2)
