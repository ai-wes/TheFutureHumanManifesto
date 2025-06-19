
import yaml
import os
from typing import Dict, Any
from dotenv import load_dotenv

class ConfigLoader:
    """Utility class for loading configuration"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        load_dotenv()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_env(self, key: str, default: str = None) -> str:
        """Get environment variable"""
        return os.getenv(key, default)

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration with environment overrides"""
        return {
            'host': self.get_env('REDIS_HOST', self.get('redis.host')),
            'port': int(self.get_env('REDIS_PORT', self.get('redis.port'))),
            'db': int(self.get_env('REDIS_DB', self.get('redis.db')))
        }

    def get_neo4j_config(self) -> Dict[str, str]:
        """Get Neo4j configuration with environment overrides"""
        return {
            'uri': self.get_env('NEO4J_URI', self.get('neo4j.uri')),
            'username': self.get_env('NEO4J_USERNAME', self.get('neo4j.username')),
            'password': self.get_env('NEO4J_PASSWORD', self.get('neo4j.password')),
            'database': self.get_env('NEO4J_DATABASE', self.get('neo4j.database'))
        }

    def get_openai_config(self) -> Dict[str, str]:
        """Get OpenAI configuration with environment overrides"""
        return {
            'api_key': self.get_env('OPENAI_API_KEY', self.get('openai.api_key')),
            'model': self.get('openai.model'),
            'max_tokens': self.get('openai.max_tokens')
        }
