# src/utils/config_loader.py
import yaml
import os
from typing import Dict, Any, Optional # Added Optional for consistency
from dotenv import load_dotenv

class ConfigLoader:
    """Utility class for loading configuration"""

    def __init__(self, config_path: Optional[str] = None): # Allow None for default
        if config_path:
            self.config_path = config_path
        else:
            # Default path, GAPS_CONFIG_PATH env var can override
            self.config_path = os.getenv("GAPS_CONFIG_PATH", "config/config.yaml")
        
        self.config = self._load_config()
        load_dotenv() # Load .env file

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        logger_ref = None # Temp logger for this critical step
        try:
            # Attempt to use project logger if available, otherwise basic print
            from custom_logging import get_logger # Try to import project logger
            logger_ref = get_logger(__name__ + ".ConfigLoader._load_config")
        except ImportError:
            pass

        try:
            if logger_ref: logger_ref.debug(f"Attempting to load config from: {os.path.abspath(self.config_path)}")
            else: print(f"ConfigLoader: Attempting to load config from: {os.path.abspath(self.config_path)}")
            
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                if logger_ref: logger_ref.info(f"Successfully loaded configuration from {self.config_path}")
                else: print(f"ConfigLoader: Successfully loaded configuration from {self.config_path}")
                return config_data if config_data else {} # Return empty dict if file is empty
        except FileNotFoundError:
            msg = f"Configuration file not found: {os.path.abspath(self.config_path)}"
            if logger_ref: logger_ref.error(msg)
            else: print(f"ERROR: {msg}")
            raise FileNotFoundError(msg)
        except yaml.YAMLError as e:
            msg = f"Invalid YAML in configuration file: {self.config_path} - {e}"
            if logger_ref: logger_ref.error(msg)
            else: print(f"ERROR: {msg}")
            raise ValueError(msg) from e
        except Exception as e:
            msg = f"Unexpected error loading configuration: {self.config_path} - {e}"
            if logger_ref: logger_ref.error(msg)
            else: print(f"ERROR: {msg}")
            raise Exception(msg) from e


    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self.config
        if not self.config and key: # If config failed to load or is empty
             # print(f"Warning: Config not loaded or empty, returning default for key '{key}'")
             return default
        try:
            for k in keys:
                if not isinstance(value, dict): # Ensure we can do value[k]
                    # print(f"Warning: Path '{key}' broken at '{k}', value is not a dict. Returning default.")
                    return default
                value = value[k]
            return value
        except KeyError:
            # print(f"Warning: Key '{key}' not found in config. Returning default.")
            return default
        except TypeError: # Handles cases where value is None and we try to index it
            # print(f"Warning: TypeError accessing key '{key}' (likely intermediate key was None). Returning default.")
            return default


    def get_env(self, key: str, default: Optional[str] = None) -> Optional[str]: # Return type can be None
        """Get environment variable"""
        return os.getenv(key, default)

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration with environment overrides"""
        port_val = self.get('redis.port')
        db_val = self.get('redis.db')
        port = 6379 # Default
        db = 0   # Default
        try:
            if port_val is not None: port = int(port_val)
            if db_val is not None: db = int(db_val)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid Redis port ('{port_val}') or DB ('{db_val}') in configuration: {e}")

        return {
            'host': self.get_env('REDIS_HOST', self.get('redis.host', 'localhost')),
            'port': int(self.get_env('REDIS_PORT', str(port))), # Env var should also be int-able
            'db': int(self.get_env('REDIS_DB', str(db)))
        }

    def get_neo4j_config(self) -> Dict[str, Optional[str]]: # Password can be None
        """Get Neo4j configuration with environment overrides"""
        return {
            'uri': self.get_env('NEO4J_URI', self.get('neo4j.uri', 'neo4j://localhost:7687')),
            'username': self.get_env('NEO4J_USERNAME', self.get('neo4j.username', 'neo4j')),
            'password': self.get_env('NEO4J_PASSWORD', self.get('neo4j.password')), # Can be None if auth disabled
            'database': self.get_env('NEO4J_DATABASE', self.get('neo4j.database', 'neo4j'))
        }

    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration with environment overrides"""
        max_tokens_val = self.get('openai.max_tokens')
        max_tokens = None
        if max_tokens_val is not None:
            try:
                max_tokens = int(max_tokens_val)
            except (ValueError, TypeError):
                # Consider logging a warning here if using project logger
                print(f"Warning: openai.max_tokens ('{max_tokens_val}') is not a valid integer. Using None.")
        
        return {
            'api_key': self.get_env('OPENAI_API_KEY', self.get('openai.api_key')),
            'model': self.get('openai.model', 'gpt-4.1-mini'), # Provide a common default
            'max_tokens': max_tokens
        }