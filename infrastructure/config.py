import os
from dataclasses import dataclass

@dataclass
class Environment:
    name: str
    account: str
    region: str
    is_production: bool
    
    @property
    def env_name(self) -> str:
        return "prod" if self.is_production else "dev"
    
    @classmethod
    def from_context(cls) -> 'Environment':
        """Create environment config from CDK context or environment variables"""
        name = os.environ.get("ENV_NAME", "dev")
        return cls(
            name=name,
            account=os.environ.get("CDK_DEFAULT_ACCOUNT", "998403024947"),
            region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
            is_production=name.lower() == "prod"
        )
