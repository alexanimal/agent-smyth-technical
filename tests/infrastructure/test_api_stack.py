import pytest
from aws_cdk import App
from aws_cdk.assertions import Template, Match
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Now import with paths relative to project root
from infrastructure.stacks.api_stack import ApiStack
from infrastructure.config import Environment

def test_api_gateway_created():
    # Arrange
    app = App()
    env_config = Environment(
        name="test",
        account="123456789012",
        region="us-east-1",
        is_production=False
    )
    
    # Act
    stack = ApiStack(app, "TestApiStack", env_config)
    template = Template.from_stack(stack)
    
    # Assert
    template.resource_count_is("AWS::ApiGateway::RestApi", 1)
    template.has_resource_properties(
        "AWS::ApiGateway::Stage",
        {
            "StageName": "dev",
            "MethodSettings": Match.array_with([
                Match.object_like({
                    "ThrottlingBurstLimit": 20,
                    "ThrottlingRateLimit": 10
                })
            ])
        }
    )
