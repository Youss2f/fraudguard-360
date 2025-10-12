# ==============================================================================
# FraudGuard 360 - Terraform Outputs
# Infrastructure resource information for external consumption
# ==============================================================================

# ==============================================================================
# VPC and Networking Outputs
# ==============================================================================

output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = module.vpc.public_subnets
}

output "database_subnet_ids" {
  description = "IDs of the database subnets"
  value       = module.vpc.database_subnets
}

output "nat_gateway_ids" {
  description = "IDs of the NAT Gateways"
  value       = module.vpc.natgw_ids
}

# ==============================================================================
# EKS Cluster Outputs
# ==============================================================================

output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_version" {
  description = "The Kubernetes version for the EKS cluster"
  value       = module.eks.cluster_version
}

output "cluster_arn" {
  description = "The Amazon Resource Name (ARN) of the cluster"
  value       = module.eks.cluster_arn
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "node_security_group_id" {
  description = "Security group ID attached to the EKS node group"
  value       = module.eks.node_security_group_id
}

output "oidc_provider_arn" {
  description = "The ARN of the OIDC Provider for EKS"
  value       = module.eks.oidc_provider_arn
}

# ==============================================================================
# Database Outputs
# ==============================================================================

output "postgresql_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = module.postgresql.db_instance_endpoint
  sensitive   = true
}

output "postgresql_port" {
  description = "RDS PostgreSQL port"
  value       = module.postgresql.db_instance_port
}

output "postgresql_database_name" {
  description = "RDS PostgreSQL database name"
  value       = module.postgresql.db_instance_name
}

output "postgresql_username" {
  description = "RDS PostgreSQL username"
  value       = module.postgresql.db_instance_username
  sensitive   = true
}

output "neo4j_endpoint" {
  description = "Neo4j database endpoint"
  value       = module.neo4j.endpoint
  sensitive   = true
}

output "neo4j_bolt_port" {
  description = "Neo4j Bolt protocol port"
  value       = 7687
}

output "neo4j_http_port" {
  description = "Neo4j HTTP port"
  value       = 7474
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
  sensitive   = true
}

output "redis_port" {
  description = "ElastiCache Redis port"
  value       = aws_elasticache_replication_group.redis.port
}

# ==============================================================================
# Kafka Outputs
# ==============================================================================

output "kafka_bootstrap_brokers" {
  description = "MSK Kafka bootstrap brokers"
  value       = module.kafka.bootstrap_brokers_tls
  sensitive   = true
}

output "kafka_zookeeper_connect_string" {
  description = "MSK Zookeeper connection string"
  value       = module.kafka.zookeeper_connect_string
  sensitive   = true
}

output "kafka_cluster_arn" {
  description = "Amazon Resource Name (ARN) of the MSK cluster"
  value       = module.kafka.cluster_arn
}

# ==============================================================================
# Load Balancer Outputs
# ==============================================================================

output "load_balancer_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.main.dns_name
}

output "load_balancer_zone_id" {
  description = "Zone ID of the load balancer"
  value       = aws_lb.main.zone_id
}

output "load_balancer_arn" {
  description = "ARN of the load balancer"
  value       = aws_lb.main.arn
}

# ==============================================================================
# Security Outputs
# ==============================================================================

output "database_security_group_id" {
  description = "Security group ID for database access"
  value       = aws_security_group.rds.id
}

output "redis_security_group_id" {
  description = "Security group ID for Redis access"
  value       = aws_security_group.redis.id
}

output "neo4j_security_group_id" {
  description = "Security group ID for Neo4j access"
  value       = aws_security_group.neo4j.id
}

# ==============================================================================
# Secret Manager Outputs
# ==============================================================================

output "postgresql_password_secret_arn" {
  description = "ARN of the PostgreSQL password secret"
  value       = aws_secretsmanager_secret.db_passwords["postgresql"].arn
  sensitive   = true
}

output "neo4j_password_secret_arn" {
  description = "ARN of the Neo4j password secret"
  value       = aws_secretsmanager_secret.db_passwords["neo4j"].arn
  sensitive   = true
}

output "redis_auth_token_secret_arn" {
  description = "ARN of the Redis auth token secret"
  value       = aws_secretsmanager_secret.db_passwords["redis"].arn
  sensitive   = true
}

# ==============================================================================
# S3 Bucket Outputs
# ==============================================================================

output "kafka_logs_bucket_name" {
  description = "Name of the S3 bucket for Kafka logs"
  value       = aws_s3_bucket.kafka_logs.id
}

output "alb_logs_bucket_name" {
  description = "Name of the S3 bucket for ALB logs"
  value       = aws_s3_bucket.alb_logs.id
}

# ==============================================================================
# KMS Key Outputs
# ==============================================================================

output "kafka_kms_key_arn" {
  description = "ARN of the KMS key for Kafka encryption"
  value       = aws_kms_key.kafka.arn
}

output "kafka_kms_key_id" {
  description = "ID of the KMS key for Kafka encryption"
  value       = aws_kms_key.kafka.key_id
}

# ==============================================================================
# Connection Information for Applications
# ==============================================================================

output "database_connection_info" {
  description = "Database connection information for applications"
  value = {
    postgresql = {
      host     = module.postgresql.db_instance_endpoint
      port     = module.postgresql.db_instance_port
      database = module.postgresql.db_instance_name
      username = module.postgresql.db_instance_username
      password_secret_arn = aws_secretsmanager_secret.db_passwords["postgresql"].arn
    }
    neo4j = {
      bolt_uri = "bolt://${module.neo4j.endpoint}:7687"
      http_uri = "http://${module.neo4j.endpoint}:7474"
      password_secret_arn = aws_secretsmanager_secret.db_passwords["neo4j"].arn
    }
    redis = {
      endpoint = aws_elasticache_replication_group.redis.primary_endpoint_address
      port     = aws_elasticache_replication_group.redis.port
      auth_token_secret_arn = aws_secretsmanager_secret.db_passwords["redis"].arn
    }
  }
  sensitive = true
}

output "messaging_connection_info" {
  description = "Messaging system connection information"
  value = {
    kafka = {
      bootstrap_brokers = module.kafka.bootstrap_brokers_tls
      zookeeper_connect = module.kafka.zookeeper_connect_string
      cluster_arn      = module.kafka.cluster_arn
    }
  }
  sensitive = true
}

# ==============================================================================
# Kubernetes Configuration
# ==============================================================================

output "kubeconfig_command" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

output "helm_values_override" {
  description = "Helm values to override with infrastructure-specific settings"
  value = {
    "global.aws.region" = var.aws_region
    "global.environment" = var.environment
    "postgresql.external.enabled" = true
    "postgresql.external.host" = module.postgresql.db_instance_endpoint
    "postgresql.external.port" = module.postgresql.db_instance_port
    "postgresql.external.database" = module.postgresql.db_instance_name
    "neo4j.external.enabled" = true
    "neo4j.external.boltUri" = "bolt://${module.neo4j.endpoint}:7687"
    "redis.external.enabled" = true
    "redis.external.host" = aws_elasticache_replication_group.redis.primary_endpoint_address
    "redis.external.port" = aws_elasticache_replication_group.redis.port
    "kafka.external.enabled" = true
    "kafka.external.bootstrapServers" = module.kafka.bootstrap_brokers_tls
  }
}