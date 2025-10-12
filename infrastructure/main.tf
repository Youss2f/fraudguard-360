# ==============================================================================
# FraudGuard 360 - Main Terraform Configuration
# Multi-cloud infrastructure provisioning with modular architecture
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
    bucket         = "fraudguard-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "fraudguard-terraform-locks"
  }
}

# ==============================================================================
# Provider Configuration
# ==============================================================================

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "FraudGuard-360"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = var.project_owner
    }
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

# ==============================================================================
# Local Values and Data Sources
# ==============================================================================

locals {
  name_prefix = "${var.project_name}-${var.environment}"
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
    Owner       = var.project_owner
  }

  # Availability zones
  azs = slice(data.aws_availability_zones.available.names, 0, 3)
}

data "aws_availability_zones" "available" {
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

data "aws_caller_identity" "current" {}

# ==============================================================================
# Random Password Generation
# ==============================================================================

resource "random_password" "db_passwords" {
  for_each = toset(["postgresql", "neo4j", "redis"])
  
  length  = 32
  special = true
}

# ==============================================================================
# VPC and Networking
# ==============================================================================

module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 6.4"

  name = "${local.name_prefix}-vpc"
  cidr = var.vpc_cidr

  azs             = local.azs
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs
  
  # Database subnets
  database_subnets                   = var.database_subnet_cidrs
  database_subnet_group_name         = "${local.name_prefix}-db-subnet-group"
  create_database_subnet_route_table = true

  # NAT Gateway
  enable_nat_gateway = true
  single_nat_gateway = var.environment == "development"
  
  # DNS
  enable_dns_hostnames = true
  enable_dns_support   = true

  # VPC Flow Logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_iam_role  = true
  create_flow_log_cloudwatch_log_group = true

  # Tags
  tags = local.common_tags

  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# ==============================================================================
# EKS Cluster
# ==============================================================================

module "eks" {
  source = "./modules/k8s"

  cluster_name    = "${local.name_prefix}-eks"
  cluster_version = var.kubernetes_version

  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets
  
  # Node groups configuration
  node_groups = var.eks_node_groups
  
  # Add-ons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  tags = local.common_tags
}

# ==============================================================================
# Managed Services
# ==============================================================================

module "kafka" {
  source = "./modules/kafka"

  cluster_name    = "${local.name_prefix}-kafka"
  kafka_version   = "2.8.1"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  broker_node_instance_type   = var.kafka_instance_type
  broker_node_storage_size    = var.kafka_storage_size
  
  enhanced_monitoring = var.environment == "production" ? "PER_TOPIC_PER_BROKER" : "DEFAULT"
  
  tags = local.common_tags
}

module "neo4j" {
  source = "./modules/neo4j"

  instance_name     = "${local.name_prefix}-neo4j"
  instance_type     = var.neo4j_instance_type
  
  vpc_id            = module.vpc.vpc_id
  subnet_ids        = module.vpc.private_subnets
  security_group_id = aws_security_group.neo4j.id
  
  database_password = random_password.db_passwords["neo4j"].result
  
  tags = local.common_tags
}

# ==============================================================================
# Security Groups
# ==============================================================================

resource "aws_security_group" "neo4j" {
  name        = "${local.name_prefix}-neo4j-sg"
  description = "Security group for Neo4j"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 7687
    to_port         = 7687
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  ingress {
    from_port       = 7474
    to_port         = 7474
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-neo4j-sg"
  })
}
