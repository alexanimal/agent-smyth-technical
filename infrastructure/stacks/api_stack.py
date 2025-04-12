# infrastructure/stacks/api_stack.py
import os

from aws_cdk import Duration
from aws_cdk import Environment as CdkEnvironment
from aws_cdk import Stack
from aws_cdk import aws_apigateway as apigw
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_ecs_patterns as ecs_patterns
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct

from infrastructure.config import Environment


class ApiStack(Stack):
    def __init__(self, scope: Construct, id: str, env_config: Environment, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create Docker image asset
        docker_image = ecr_assets.DockerImageAsset(
            self,
            f"{id}-DockerImage",
            directory=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # Project root
        )

        # Use NetworkLoadBalancedFargateService with proper health check config
        fargate_service = ecs_patterns.NetworkLoadBalancedFargateService(
            self,
            f"{id}-Service",
            task_image_options=ecs_patterns.NetworkLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_docker_image_asset(docker_image),
                container_port=8003,
                environment={
                    "ENVIRONMENT": env_config.env_name,
                },
                secrets={
                    "OPENAI_API_KEY": ecs.Secret.from_secrets_manager(
                        # Use the existing secret named AgentSmyth
                        secretsmanager.Secret.from_secret_name_v2(
                            self, f"{id}-OpenAiApiKey", "AgentSmyth"
                        ),
                        # Extract the field named OPENAI_API_KEY from the secret's JSON value
                        "OPENAI_API_KEY",
                    ),
                    "SENTRY_DSN": ecs.Secret.from_secrets_manager(
                        # Use the existing secret named AgentSmyth
                        secretsmanager.Secret.from_secret_name_v2(
                            self, f"{id}-SentryDsn", "AgentSmyth"
                        ),
                        "SENTRY_DSN",
                    ),
                },
            ),
            desired_count=1,
            memory_limit_mib=32768,
            cpu=8192,
            # Listener port for NLB
            listener_port=80,
            # Add health check grace period - give app time to start
            health_check_grace_period=Duration.seconds(300),  # 5 minutes
            # Add deployment configuration to fix the minHealthyPercent warning
            min_healthy_percent=100,  # Keep at least one task running during deployment
            max_healthy_percent=200,  # Allow starting a new task before stopping the old one
        )

        # Configure the target group's health check
        fargate_service.target_group.configure_health_check(
            port="8003",
            protocol=elbv2.Protocol.HTTP,
            path="/health",  # Assuming your app has a health endpoint
            healthy_threshold_count=2,
            unhealthy_threshold_count=5,  # More tolerant
            timeout=Duration.seconds(10),
            interval=Duration.seconds(30),
        )

        # Create VPC Link
        vpc_link = apigw.VpcLink(
            self,
            f"{id}-VpcLink",
            targets=[fargate_service.load_balancer],
            description="VPC Link for API Gateway to Fargate NLB",
        )

        # Create API Gateway pointing to NLB via VPC Link
        api = apigw.RestApi(
            self,
            f"{id}-ApiGateway",
            rest_api_name=f"{id}-RestApi",
            description=f"API Gateway for {id} service",
            deploy_options=apigw.StageOptions(
                stage_name=env_config.env_name, throttling_burst_limit=20, throttling_rate_limit=10
            ),
        )

        # Create API Gateway NLB Integration
        nlb_integration = apigw.Integration(
            type=apigw.IntegrationType.HTTP_PROXY,
            integration_http_method="ANY",
            options=apigw.IntegrationOptions(
                connection_type=apigw.ConnectionType.VPC_LINK,
                vpc_link=vpc_link,
                passthrough_behavior=apigw.PassthroughBehavior.WHEN_NO_MATCH,
            ),
            uri=f"http://{fargate_service.load_balancer.load_balancer_dns_name}",
        )

        # Add Proxy Resource to API Gateway
        api.root.add_proxy(default_integration=nlb_integration, any_method=True)

        aws_env = CdkEnvironment(
            account=env_config.account or os.environ.get("CDK_DEFAULT_ACCOUNT", ""),
            region=env_config.region or os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
        )
