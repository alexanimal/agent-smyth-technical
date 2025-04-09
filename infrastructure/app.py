#!/usr/bin/env python3
import os

from aws_cdk import App
from aws_cdk import Environment as CdkEnvironment

from infrastructure.config import Environment
from infrastructure.stacks.api_stack import ApiStack

app = App()

# Load environment configuration
env_config = Environment.from_context()

# Define AWS environment
aws_env = CdkEnvironment(account=env_config.account, region=env_config.region)

# Create stacks with naming convention based on environment
base_id = "AgentSmyth"
env_suffix = env_config.env_name.capitalize()

# Create stacks
api = ApiStack(app, f"{base_id}Api{env_suffix}", env_config, env=aws_env)

app.synth()
