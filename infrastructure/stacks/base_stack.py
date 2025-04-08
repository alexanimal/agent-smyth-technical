from aws_cdk import Stack, Tags
from constructs import Construct
from ..config import Environment

class BaseStack(Stack):
    def __init__(self, scope: Construct, id: str, env_config: Environment, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        
        # Apply tags to all resources in the stack
        Tags.of(self).add("Project", "AgentSmyth")
        Tags.of(self).add("Environment", env_config.env_name)
        Tags.of(self).add("ManagedBy", "CDK")
