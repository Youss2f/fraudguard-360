# ==============================================================================
# FraudGuard 360 - Development Environment Main Configuration
# Development-specific Terraform configuration
# ==============================================================================

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }

  backend "s3" {
    bucket         = "fraudguard-360-terraform-state-dev"
    key            = "dev/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "fraudguard-360-terraform-locks"
    encrypt        = true
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = var.common_tags
  }
}

# Data source for EKS cluster
data "aws_eks_cluster" "cluster" {
  depends_on = [module.infrastructure]
  name       = module.infrastructure.eks_cluster_name
}

data "aws_eks_cluster_auth" "cluster" {
  depends_on = [module.infrastructure]
  name       = module.infrastructure.eks_cluster_name
}

# Configure Kubernetes Provider
provider "kubernetes" {
  host                   = data.aws_eks_cluster.cluster.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}

# Configure Helm Provider
provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.cluster.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
    token                  = data.aws_eks_cluster_auth.cluster.token
  }
}

# Main Infrastructure Module
module "infrastructure" {
  source = "../../"

  # Environment Configuration
  environment = var.environment
  aws_region  = var.aws_region

  # VPC Configuration
  vpc_cidr                 = var.vpc_cidr
  availability_zones       = var.availability_zones
  public_subnet_cidrs      = var.public_subnet_cidrs
  private_subnet_cidrs     = var.private_subnet_cidrs
  database_subnet_cidrs    = var.database_subnet_cidrs
  enable_nat_gateway       = var.enable_nat_gateway
  single_nat_gateway       = var.single_nat_gateway
  enable_flow_logs         = var.enable_flow_logs

  # EKS Configuration
  cluster_version                      = var.cluster_version
  cluster_endpoint_private_access      = var.cluster_endpoint_private_access
  cluster_endpoint_public_access       = var.cluster_endpoint_public_access
  cluster_endpoint_public_access_cidrs = var.cluster_endpoint_public_access_cidrs
  node_groups                          = var.node_groups
  enable_cluster_autoscaler            = var.enable_cluster_autoscaler
  enable_metrics_server                = var.enable_metrics_server

  # RDS Configuration
  rds_instance_class                   = var.rds_instance_class
  rds_allocated_storage                = var.rds_allocated_storage
  rds_max_allocated_storage            = var.rds_max_allocated_storage
  rds_backup_retention_period          = var.rds_backup_retention_period
  rds_deletion_protection              = var.rds_deletion_protection
  rds_skip_final_snapshot              = var.rds_skip_final_snapshot
  rds_performance_insights_enabled     = var.rds_performance_insights_enabled

  # ElastiCache Configuration
  redis_node_type                      = var.redis_node_type
  redis_num_cache_nodes                = var.redis_num_cache_nodes
  redis_parameter_group_name           = var.redis_parameter_group_name
  redis_engine_version                 = var.redis_engine_version
  redis_at_rest_encryption_enabled     = var.redis_at_rest_encryption_enabled
  redis_transit_encryption_enabled     = var.redis_transit_encryption_enabled

  # MSK Configuration
  kafka_instance_type        = var.kafka_instance_type
  kafka_ebs_volume_size      = var.kafka_ebs_volume_size
  kafka_broker_count         = var.kafka_broker_count
  kafka_client_subnets_count = var.kafka_client_subnets_count

  # Neo4j Configuration
  neo4j_instance_type       = var.neo4j_instance_type
  neo4j_min_size            = var.neo4j_min_size
  neo4j_max_size            = var.neo4j_max_size
  neo4j_desired_capacity    = var.neo4j_desired_capacity
  neo4j_heap_initial_size   = var.neo4j_heap_initial_size
  neo4j_heap_max_size       = var.neo4j_heap_max_size
  neo4j_pagecache_size      = var.neo4j_pagecache_size
  neo4j_ebs_volume_size     = var.neo4j_ebs_volume_size
  neo4j_ebs_volume_type     = var.neo4j_ebs_volume_type
  neo4j_ebs_iops            = var.neo4j_ebs_iops

  # Security Configuration
  allowed_cidr_blocks = var.allowed_cidr_blocks

  # Monitoring Configuration
  enable_cloudwatch_logs = var.enable_cloudwatch_logs
  log_retention_days     = var.log_retention_days

  # Load Balancer Configuration
  enable_deletion_protection = var.enable_deletion_protection

  # Common Tags
  common_tags = var.common_tags
}

# Monitoring Module (deployed after infrastructure)
module "monitoring" {
  source = "../../modules/monitoring"
  
  depends_on = [module.infrastructure]

  # Environment Configuration
  environment = var.environment

  # Grafana Configuration (development settings)
  grafana_admin_user     = "admin"
  grafana_admin_password = "fraudguard-dev-2024"
  grafana_service_type   = "LoadBalancer"  # For easy access in dev

  # Resource Configuration (smaller for dev)
  prometheus_replicas = 1
  grafana_replicas    = 1
  alertmanager_replicas = 1

  # Resource Limits (smaller for dev)
  prometheus_cpu_limit    = "1000m"
  prometheus_memory_limit = "2Gi"
  grafana_cpu_limit       = "500m"
  grafana_memory_limit    = "512Mi"

  # Alert Configuration (development)
  alert_email_configs = [
    {
      to      = "dev-team@fraudguard360.com"
      subject = "[DEV] FraudGuard 360 Alert: {{ .GroupLabels.alertname }}"
      body    = "Development Environment Alert\nAlert: {{ .GroupLabels.alertname }}\nSeverity: {{ .CommonLabels.severity }}\nDescription: {{ .CommonAnnotations.description }}"
    }
  ]

  # Common Tags
  common_tags = var.common_tags
}