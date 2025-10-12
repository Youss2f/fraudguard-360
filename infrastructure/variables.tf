# ==============================================================================
# FraudGuard 360 - Terraform Variables
# Configurable infrastructure parameters for all environments
# ==============================================================================

# ==============================================================================
# General Configuration
# ==============================================================================

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "fraudguard-360"
}

variable "environment" {
  description = "Environment name (development, staging, production)"
  type        = string
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be one of: development, staging, production."
  }
}

variable "project_owner" {
  description = "Owner of the project for tagging"
  type        = string
  default     = "FraudGuard Team"
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

# ==============================================================================
# Networking Configuration
# ==============================================================================

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)  
  default     = ["10.0.11.0/24", "10.0.12.0/24", "10.0.13.0/24"]
}

variable "database_subnet_cidrs" {
  description = "CIDR blocks for database subnets"
  type        = list(string)
  default     = ["10.0.21.0/24", "10.0.22.0/24", "10.0.23.0/24"]
}

# ==============================================================================
# Kubernetes Configuration
# ==============================================================================

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "eks_node_groups" {
  description = "EKS node group configurations"
  type = map(object({
    instance_types = list(string)
    capacity_type  = string
    scaling_config = object({
      desired_size = number
      max_size     = number
      min_size     = number
    })
    update_config = object({
      max_unavailable_percentage = number
    })
    labels = map(string)
    taints = list(object({
      key    = string
      value  = string
      effect = string
    }))
  }))
  default = {
    general = {
      instance_types = ["t3.large"]
      capacity_type  = "ON_DEMAND"
      scaling_config = {
        desired_size = 3
        max_size     = 10
        min_size     = 1
      }
      update_config = {
        max_unavailable_percentage = 25
      }
      labels = {
        role = "general"
      }
      taints = []
    }
    compute = {
      instance_types = ["c5.xlarge"]
      capacity_type  = "SPOT"
      scaling_config = {
        desired_size = 2
        max_size     = 20
        min_size     = 0
      }
      update_config = {
        max_unavailable_percentage = 25
      }
      labels = {
        role = "compute"
      }
      taints = [
        {
          key    = "compute"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
}

# ==============================================================================
# Database Configuration
# ==============================================================================

variable "rds_instance_class" {
  description = "RDS instance class for PostgreSQL"
  type        = string
  default     = "db.r6g.large"
}

variable "rds_allocated_storage" {
  description = "Initial allocated storage for RDS in GB"
  type        = number
  default     = 100
}

variable "rds_max_allocated_storage" {
  description = "Maximum allocated storage for RDS in GB"
  type        = number
  default     = 1000
}

variable "neo4j_instance_type" {
  description = "EC2 instance type for Neo4j"
  type        = string
  default     = "r6i.xlarge"
}

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.r6g.large"
}

# ==============================================================================
# Kafka Configuration
# ==============================================================================

variable "kafka_instance_type" {
  description = "MSK broker instance type"
  type        = string
  default     = "kafka.m5.large"
}

variable "kafka_storage_size" {
  description = "Storage size per broker in GB"
  type        = number
  default     = 100
}

# ==============================================================================
# Monitoring Configuration
# ==============================================================================

variable "enable_monitoring" {
  description = "Enable monitoring stack (Prometheus, Grafana)"
  type        = bool
  default     = true
}

variable "enable_logging" {
  description = "Enable centralized logging (ELK stack)"
  type        = bool
  default     = true
}

variable "enable_tracing" {
  description = "Enable distributed tracing (Jaeger)"
  type        = bool
  default     = false
}

# ==============================================================================
# Security Configuration
# ==============================================================================

variable "enable_network_policies" {
  description = "Enable Kubernetes network policies"
  type        = bool
  default     = true
}

variable "enable_pod_security_policies" {
  description = "Enable Kubernetes pod security policies"
  type        = bool
  default     = true
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access load balancer"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# ==============================================================================
# Environment-Specific Overrides
# ==============================================================================

variable "environment_configs" {
  description = "Environment-specific configuration overrides"
  type = map(object({
    node_desired_size    = number
    node_max_size       = number
    rds_instance_class  = string
    kafka_instance_type = string
    enable_nat_gateway  = bool
    backup_retention    = number
  }))
  default = {
    development = {
      node_desired_size    = 2
      node_max_size       = 5
      rds_instance_class  = "db.t3.medium"
      kafka_instance_type = "kafka.t3.small"
      enable_nat_gateway  = false
      backup_retention    = 7
    }
    staging = {
      node_desired_size    = 3
      node_max_size       = 8
      rds_instance_class  = "db.r6g.large"
      kafka_instance_type = "kafka.m5.large"
      enable_nat_gateway  = true
      backup_retention    = 14
    }
    production = {
      node_desired_size    = 5
      node_max_size       = 20
      rds_instance_class  = "db.r6g.2xlarge"
      kafka_instance_type = "kafka.m5.xlarge"
      enable_nat_gateway  = true
      backup_retention    = 30
    }
  }
}