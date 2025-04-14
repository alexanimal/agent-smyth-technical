# infrastructure/stacks/api_stack.py
import os

from aws_cdk import Duration
from aws_cdk import Environment as CdkEnvironment
from aws_cdk import Stack
from aws_cdk import aws_apigateway as apigw
from aws_cdk import aws_certificatemanager as acm
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_ecs_patterns as ecs_patterns
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_route53 as route53
from aws_cdk import aws_route53_targets as targets
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct

from infrastructure.config import Environment


class ApiStack(Stack):
    def __init__(self, scope: Construct, id: str, env_config: Environment, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        certificate_arn = (
            "arn:aws:acm:us-east-1:998403024947:certificate/03279c6f-9a33-441b-aa00-8defb86224ab"
        )

        # Look up the existing certificate
        certificate = acm.Certificate.from_certificate_arn(
            self, f"{id}-Certificate", certificate_arn=certificate_arn
        )
        # --- End Certificate Setup ---

        # Create Docker image asset
        docker_image = ecr_assets.DockerImageAsset(
            self,
            f"{id}-DockerImage",
            directory=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # Project root
            # Exclude files not needed for the Docker image build to prevent errors and optimize
            exclude=[
                ".git*",
                ".venv",
                "cdk.out",
                "tests",
                "*.pem",
                "*.key",
                "*.cnf",
                ".pytest_cache",
                "__pycache__",
                "*.pyc",
                "README.md",
                ".vscode",
                "*.md",
            ],
        )

        # Use NetworkLoadBalancedFargateService with proper health check config
        fargate_service = ecs_patterns.NetworkLoadBalancedFargateService(
            self,
            f"{id}-Service",
            task_image_options=ecs_patterns.NetworkLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_docker_image_asset(docker_image),
                container_port=8003,
                environment={
                    "ENVIRONMENT": "production",
                    "LANGSMITH_TRACING": "true",
                    "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
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
                    "API_KEY": ecs.Secret.from_secrets_manager(
                        # Use the existing secret named AgentSmyth
                        secretsmanager.Secret.from_secret_name_v2(
                            self, f"{id}-ApiKey", "AgentSmyth"
                        ),
                        "API_KEY",
                    ),
                    "LANGSMITH_API_KEY": ecs.Secret.from_secrets_manager(
                        # Use the existing secret named AgentSmyth
                        secretsmanager.Secret.from_secret_name_v2(
                            self, f"{id}-LangsmithApiKey", "AgentSmyth"
                        ),
                        "LANGSMITH_API_KEY",
                    ),
                    "LANGSMITH_PROJECT_NAME": ecs.Secret.from_secrets_manager(
                        # Use the existing secret named AgentSmyth
                        secretsmanager.Secret.from_secret_name_v2(
                            self, f"{id}-LangsmithProjectName", "AgentSmyth"
                        ),
                        "LANGSMITH_PROJECT_NAME",
                    ),
                },
            ),
            desired_count=2,
            min_healthy_percent=100,
            max_healthy_percent=200,
            memory_limit_mib=8192,  # 8GB is sufficient and more likely to be available
            cpu=2048,  # 2 vCPU is reasonable for this workload
            # Listener port for NLB
            listener_port=80,
            # Add health check grace period - give app time to start
            health_check_grace_period=Duration.seconds(300),  # 5 minutes
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

        # Explicitly Add Security Group Rule
        fargate_service.service.connections.allow_from(
            # Source: The Network Load Balancer created by the pattern
            fargate_service.load_balancer,
            # Port Range: The specific container port
            ec2.Port.tcp(8003),
            # Description for the security group rule
            f"Allow inbound traffic from NLB on port 8003 for {id}",
        )

        # --- Add HTTPS/TLS Listener to the NLB ---
        https_listener = fargate_service.load_balancer.add_listener(
            f"{id}-HttpsListener",
            port=443,
            protocol=elbv2.Protocol.TLS,  # Use TLS protocol for NLB HTTPS
            certificates=[certificate],  # Reference the ACM certificate
            # Forward traffic to the same target group as the HTTP listener
            default_target_groups=[fargate_service.target_group],
        )
        # --- End Listener Addition ---

        # Create VPC Link
        vpc_link = apigw.VpcLink(
            self,
            f"{id}-VpcLink",
            targets=[fargate_service.load_balancer],
            description="VPC Link for API Gateway to Fargate NLB",
        )

        # Create API Gateway pointing to NLB via VPC Link
        hosted_zone_name = "alexanimal.com"  # Define your domain name
        api_domain_name = f"agent-smyth-api.{hosted_zone_name}"
        api = apigw.RestApi(
            self,
            f"{id}-ApiGateway",
            rest_api_name=f"{id}-RestApi",
            description=f"API Gateway for {id} service",
            deploy_options=apigw.StageOptions(
                stage_name=env_config.env_name, throttling_burst_limit=20, throttling_rate_limit=10
            ),
            # Add Custom Domain Configuration
            domain_name=apigw.DomainNameOptions(
                domain_name=api_domain_name,
                certificate=certificate,
                endpoint_type=apigw.EndpointType.REGIONAL,  # Use REGIONAL for NLB integration
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

        # Define the /chat resource
        chat_resource = api.root.add_resource("chat")

        # Add POST method to /chat
        chat_resource.add_method(
            "POST",
            nlb_integration,
        )

        # Define the /chat/stream resource
        chat_stream_resource = chat_resource.add_resource("stream")

        # Add POST method to /chat/stream
        chat_stream_resource.add_method(
            "POST",
            nlb_integration,
        )

        # --- Route53 Records ---
        hosted_zone_name = "alexanimal.com"  # Define your domain name

        # Look up the existing Hosted Zone
        hosted_zone = route53.HostedZone.from_lookup(
            self, f"{id}-HostedZone", domain_name=hosted_zone_name
        )

        # Create an A record for the API Gateway
        route53.ARecord(
            self,
            f"{id}-ApiGatewayAliasRecord",
            zone=hosted_zone,
            record_name="agent-smyth-api",  # Creates agent-smyth-api.alexanimal.com
            target=route53.RecordTarget.from_alias(targets.ApiGateway(api)),  # type: ignore[arg-type]
        )

        # Create an A record for the NLB directly
        route53.ARecord(
            self,
            f"{id}-NlbAliasRecord",
            zone=hosted_zone,
            record_name="agent-smyth-lb",  # Creates agent-smyth-lb.alexanimal.com
            target=route53.RecordTarget.from_alias(targets.LoadBalancerTarget(fargate_service.load_balancer)),  # type: ignore[arg-type]
        )
        # --- End Route53 Records ---

        aws_env = CdkEnvironment(
            account=env_config.account or os.environ.get("CDK_DEFAULT_ACCOUNT", ""),
            region=env_config.region or os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
        )
