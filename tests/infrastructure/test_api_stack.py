import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from aws_cdk import App, Stack
from aws_cdk import aws_route53 as route53
from aws_cdk.assertions import Match, Template

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from infrastructure.config import Environment

# Now import with paths relative to project root
from infrastructure.stacks.api_stack import ApiStack

# Define the target string for patching route53.HostedZone.from_lookup
# This needs to target where it's *used*, which is inside the api_stack module
HOSTED_ZONE_LOOKUP_TARGET = "infrastructure.stacks.api_stack.route53.HostedZone.from_lookup"


# Use the patch decorator to mock the lookup call for the duration of the test
@patch(HOSTED_ZONE_LOOKUP_TARGET)
def test_api_gateway_created(mock_from_lookup: MagicMock):
    """
    Tests that the ApiStack synthesizes an API Gateway and Stage correctly,
    mocking the Route53 hosted zone lookup to avoid external calls.
    """
    # Arrange
    # Use a single App for both the test stack and the mock resource stack
    app = App()

    # Create a minimal Stack within the *same App* to host the mock HostedZone
    mock_zone_stack = Stack(app, "MockZoneStack")  # Use 'app' here
    mock_hosted_zone = route53.HostedZone(
        mock_zone_stack,
        "MockHostedZone",
        zone_name="alexanimal.com",  # Provide the required zone_name
    )
    mock_from_lookup.return_value = mock_hosted_zone

    # Define a test environment configuration. Note: The CDK Stack `env` property
    # (account/region) would normally be needed for context lookups, but we are
    # mocking the lookup, so it's not strictly required for this mock to work.
    # However, the env_config object itself IS used for the StageName.
    env_config = Environment(
        name="dev",  # Use 'dev' to match the expected StageName property below
        account="123456789012",  # Dummy account
        region="us-east-1",  # Dummy region
        is_production=False,
    )

    # Act
    # Instantiate the stack. The call to route53.HostedZone.from_lookup inside
    # ApiStack's __init__ will now use our mock.
    stack = ApiStack(app, "TestApiStack", env_config)
    template = Template.from_stack(stack)

    # Assert
    # Verify the lookup was called once with expected arguments
    mock_from_lookup.assert_called_once_with(
        stack,  # scope
        "TestApiStack-HostedZone",  # construct ID
        domain_name="alexanimal.com",  # lookup arguments
    )

    # Assert that the RestApi resource was created (as per the original test)
    template.resource_count_is("AWS::ApiGateway::RestApi", 1)

    # Assert that the Stage was created with the correct name and throttling properties
    # (as per the original test, using the 'dev' stage name from env_config)
    template.has_resource_properties(
        "AWS::ApiGateway::Stage",
        {
            "StageName": "dev",
            "MethodSettings": Match.array_with(
                [Match.object_like({"ThrottlingBurstLimit": 20, "ThrottlingRateLimit": 10})]
            ),
        },
    )
