# ==============================================================================
# FraudGuard 360 - Development Environment Variables
# Variable definitions for development environment
# ==============================================================================

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "common_tags" {
  description = "Common tags to be applied to all resources"
  type        = map(string)
  default     = {}
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
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

variable "enable_nat_gateway" {
  description = "Enable NAT gateway"
  type        = bool
  default     = true
}

variable "single_nat_gateway" {
  description = "Use single NAT gateway"
  type        = bool
  default     = true
}

variable "enable_flow_logs" {
  description = "Enable VPC flow logs"
  type        = bool
  default     = false
}

# EKS Configuration
variable "cluster_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "cluster_endpoint_private_access" {
  description = "Enable private API server endpoint"
  type        = bool
  default     = true
}

variable "cluster_endpoint_public_access" {
  description = "Enable public API server endpoint"
  type        = bool
  default     = true
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "CIDR blocks for public API server endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "node_groups" {
  description = "EKS node groups configuration"
  type = map(object({
    name           = string
    instance_types = list(string)
    ami_type       = string
    capacity_type  = string
    disk_size      = number
    desired_size   = number
    max_size       = number
    min_size       = number
  }))
  default = {}
}

variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler"
  type        = bool
  default     = true
}

variable "enable_metrics_server" {
  description = "Enable metrics server"
  type        = bool
  default     = true
}

# RDS Configuration
variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage"
  type        = number
  default     = 20
}

variable "rds_max_allocated_storage" {
  description = "RDS max allocated storage"
  type        = number
  default     = 100
}

variable "rds_backup_retention_period" {
  description = "RDS backup retention period"
  type        = number
  default     = 3
}

variable "rds_deletion_protection" {
  description = "RDS deletion protection"
  type        = bool
  default     = false
}

variable "rds_skip_final_snapshot" {
  description = "Skip final snapshot"
  type        = bool
  default     = true
}

variable "rds_performance_insights_enabled" {
  description = "Enable Performance Insights"
  type        = bool
  default     = false
}

# ElastiCache Configuration
variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes"
  type        = number
  default     = 1
}

variable "redis_parameter_group_name" {
  description = "Redis parameter group name"
  type        = string
  default     = "default.redis7"
}

variable "redis_engine_version" {
  description = "Redis engine version"
  type        = string
  default     = "7.0"
}

variable "redis_at_rest_encryption_enabled" {
  description = "Enable at-rest encryption"
  type        = bool
  default     = false
}

variable "redis_transit_encryption_enabled" {
  description = "Enable transit encryption"
  type        = bool
  default     = false
}

# MSK Configuration
variable "kafka_instance_type" {
  description = "Kafka instance type"
  type        = string
  default     = "kafka.t3.small"
}

variable "kafka_ebs_volume_size" {
  description = "Kafka EBS volume size"
  type        = number
  default     = 10
}

variable "kafka_broker_count" {
  description = "Number of Kafka brokers"
  type        = number
  default     = 2
}

variable "kafka_client_subnets_count" {
  description = "Number of client subnets"
  type        = number
  default     = 2
}

# Neo4j Configuration
variable "neo4j_instance_type" {
  description = "Neo4j instance type"
  type        = string
  default     = "t3.medium"
}

variable "neo4j_min_size" {
  description = "Neo4j minimum instances"
  type        = number
  default     = 1
}

variable "neo4j_max_size" {
  description = "Neo4j maximum instances"
  type        = number
  default     = 2
}

variable "neo4j_desired_capacity" {
  description = "Neo4j desired capacity"
  type        = number
  default     = 1
}

variable "neo4j_heap_initial_size" {
  description = "Neo4j initial heap size"
  type        = string
  default     = "1G"
}

variable "neo4j_heap_max_size" {
  description = "Neo4j maximum heap size"
  type        = string
  default     = "1G"
}

variable "neo4j_pagecache_size" {
  description = "Neo4j page cache size"
  type        = string
  default     = "512M"
}

variable "neo4j_ebs_volume_size" {
  description = "Neo4j EBS volume size"
  type        = number
  default     = 50
}

variable "neo4j_ebs_volume_type" {
  description = "Neo4j EBS volume type"
  type        = string
  default     = "gp3"
}

variable "neo4j_ebs_iops" {
  description = "Neo4j EBS IOPS"
  type        = number
  default     = 3000
}

# Security Configuration
variable "allowed_cidr_blocks" {
  description = "Allowed CIDR blocks"
  type        = list(string)
  default     = ["10.0.0.0/16"]
}

# Monitoring Configuration
variable "enable_cloudwatch_logs" {
  description = "Enable CloudWatch logs"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "Log retention days"
  type        = number
  default     = 7
}

# Load Balancer Configuration
variable "enable_deletion_protection" {
  description = "Enable deletion protection"
  type        = bool
  default     = false
}