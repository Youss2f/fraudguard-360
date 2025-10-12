# ==============================================================================
# FraudGuard 360 - Development Environment Outputs
# Output values from development environment
# ==============================================================================

output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

# VPC Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = module.infrastructure.vpc_id
}

output "vpc_cidr_block" {
  description = "VPC CIDR block"
  value       = module.infrastructure.vpc_cidr_block
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = module.infrastructure.public_subnet_ids
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = module.infrastructure.private_subnet_ids
}

output "database_subnet_ids" {
  description = "Database subnet IDs"
  value       = module.infrastructure.database_subnet_ids
}

# EKS Outputs
output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.infrastructure.eks_cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.infrastructure.eks_cluster_endpoint
}

output "eks_cluster_version" {
  description = "EKS cluster version"
  value       = module.infrastructure.eks_cluster_version
}

output "eks_cluster_certificate_authority_data" {
  description = "EKS cluster certificate authority data"
  value       = module.infrastructure.eks_cluster_certificate_authority_data
  sensitive   = true
}

output "eks_node_groups" {
  description = "EKS node groups information"
  value       = module.infrastructure.eks_node_groups
}

# RDS Outputs
output "rds_endpoint" {
  description = "RDS endpoint"
  value       = module.infrastructure.rds_endpoint
}

output "rds_port" {
  description = "RDS port"
  value       = module.infrastructure.rds_port
}

output "rds_database_name" {
  description = "RDS database name"
  value       = module.infrastructure.rds_database_name
}

output "rds_master_username" {
  description = "RDS master username"
  value       = module.infrastructure.rds_master_username
  sensitive   = true
}

output "rds_secrets_manager_secret_arn" {
  description = "RDS Secrets Manager secret ARN"
  value       = module.infrastructure.rds_secrets_manager_secret_arn
}

# ElastiCache Outputs
output "redis_endpoint" {
  description = "Redis endpoint"
  value       = module.infrastructure.redis_endpoint
}

output "redis_port" {
  description = "Redis port"
  value       = module.infrastructure.redis_port
}

# MSK Outputs
output "kafka_bootstrap_brokers" {
  description = "Kafka bootstrap brokers"
  value       = module.infrastructure.kafka_bootstrap_brokers
}

output "kafka_bootstrap_brokers_tls" {
  description = "Kafka bootstrap brokers TLS"
  value       = module.infrastructure.kafka_bootstrap_brokers_tls
}

output "kafka_zookeeper_connect_string" {
  description = "Kafka Zookeeper connect string"
  value       = module.infrastructure.kafka_zookeeper_connect_string
}

# Neo4j Outputs
output "neo4j_endpoint" {
  description = "Neo4j endpoint"
  value       = module.infrastructure.neo4j_endpoint
}

output "neo4j_bolt_uri" {
  description = "Neo4j Bolt URI"
  value       = module.infrastructure.neo4j_bolt_uri
}

output "neo4j_http_uri" {
  description = "Neo4j HTTP URI"
  value       = module.infrastructure.neo4j_http_uri
}

# Monitoring Outputs
output "monitoring_namespace" {
  description = "Monitoring namespace"
  value       = module.monitoring.monitoring_namespace
}

output "prometheus_endpoint" {
  description = "Prometheus endpoint"
  value       = module.monitoring.prometheus_endpoint
}

output "grafana_endpoint" {
  description = "Grafana endpoint"
  value       = module.monitoring.grafana_endpoint
}

output "grafana_admin_credentials" {
  description = "Grafana admin credentials"
  value       = module.monitoring.grafana_admin_credentials
  sensitive   = true
}

output "alertmanager_endpoint" {
  description = "AlertManager endpoint"
  value       = module.monitoring.alertmanager_endpoint
}

# Connection Information
output "connection_info" {
  description = "Connection information for all services"
  value = {
    eks = {
      cluster_name = module.infrastructure.eks_cluster_name
      endpoint     = module.infrastructure.eks_cluster_endpoint
    }
    rds = {
      endpoint     = module.infrastructure.rds_endpoint
      port         = module.infrastructure.rds_port
      database     = module.infrastructure.rds_database_name
      secret_arn   = module.infrastructure.rds_secrets_manager_secret_arn
    }
    redis = {
      endpoint = module.infrastructure.redis_endpoint
      port     = module.infrastructure.redis_port
    }
    kafka = {
      bootstrap_brokers     = module.infrastructure.kafka_bootstrap_brokers
      bootstrap_brokers_tls = module.infrastructure.kafka_bootstrap_brokers_tls
      zookeeper_connect     = module.infrastructure.kafka_zookeeper_connect_string
    }
    neo4j = {
      endpoint  = module.infrastructure.neo4j_endpoint
      bolt_uri  = module.infrastructure.neo4j_bolt_uri
      http_uri  = module.infrastructure.neo4j_http_uri
    }
    monitoring = {
      prometheus   = module.monitoring.prometheus_endpoint
      grafana      = module.monitoring.grafana_endpoint
      alertmanager = module.monitoring.alertmanager_endpoint
    }
  }
  sensitive = true
}

# Quick Access Commands
output "kubectl_config_command" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.infrastructure.eks_cluster_name}"
}

output "grafana_port_forward_command" {
  description = "Command to port forward Grafana"
  value       = "kubectl port-forward -n ${module.monitoring.monitoring_namespace} svc/${module.monitoring.grafana_service_name} 3000:3000"
}

output "prometheus_port_forward_command" {
  description = "Command to port forward Prometheus"
  value       = "kubectl port-forward -n ${module.monitoring.monitoring_namespace} svc/${module.monitoring.prometheus_service_name} 9090:9090"
}