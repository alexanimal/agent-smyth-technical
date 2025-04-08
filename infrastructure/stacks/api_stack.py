# infrastructure/stacks/api_stack.py
from aws_cdk import (
    Stack,
    aws_lambda as lambda_,
    aws_apigateway as apigw,
    aws_secretsmanager as secretsmanager,
    aws_ecr_assets as ecr_assets,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    Duration,
    Environment as CdkEnvironment
)
from constructs import Construct
from infrastructure.config import Environment
import os


class ApiStack(Stack):
    def __init__(self, scope: Construct, id: str, env_config: Environment, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        
        # Create secrets in CDK
        openai_secret = secretsmanager.Secret(
            self, f"{id}-OpenAISecret",
            secret_name="OpenAI",
            description="OpenAI API Key",
        )
        
        # Create Docker image asset
        docker_image = ecr_assets.DockerImageAsset(
            self, f"{id}-DockerImage",
            directory=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # Project root
        )
        
        # Use Fargate service
        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, f"{id}-Service",
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_docker_image_asset(docker_image),
                container_port=8002,
                environment={
                    "ENVIRONMENT": env_config.env_name,
                },
                secrets={
                    "OPENAI_API_KEY": ecs.Secret.from_secrets_manager(
                        secretsmanager.Secret.from_secret_name_v2(
                            self, f"{id}-OpenAISecretImport", "OpenAI"
                        ),
                        "ApiKey"
                    ),
                },
            ),
            desired_count=2,
            memory_limit_mib=2048,
            cpu=1024,
        )
        
        # Define Lambda for FastAPI
        handler = lambda_.Function(
            self, f"{id}-Lambda",
            runtime=lambda_.Runtime.PYTHON_3_10,
            code=lambda_.Code.from_asset("src"),
            handler="main.handler",
            environment={
                "ENVIRONMENT": env_config.env_name,
                "OPENAI_API_KEY": openai_secret.secret_value_from_json("ApiKey").to_string(),
            },
            memory_size=1024,
            timeout=Duration.seconds(30),
        )
        
        # Create API Gateway with properly applied throttling settings
        api = apigw.LambdaRestApi(
            self, f"{id}-API",
            handler=handler,
            proxy=True,
            deploy_options=apigw.StageOptions(
                stage_name=env_config.env_name,
                # Apply throttling using method_options instead:
                method_options={
                    "/*/*": apigw.MethodDeploymentOptions(
                        throttling_burst_limit=20,
                        throttling_rate_limit=10
                    )
                }
            )
        )

        aws_env = CdkEnvironment(
            account=env_config.account or os.environ.get("CDK_DEFAULT_ACCOUNT", ""),
            region=env_config.region or os.environ.get("CDK_DEFAULT_REGION", "us-east-1")
        )