# ==============================================================================
# FraudGuard 360 - Neo4j Module Variables
# Input variables for Neo4j database configuration
# ==============================================================================

variable "instance_name" {
  description = "Name of the Neo4j instance"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type for Neo4j"
  type        = string
  default     = "r6i.xlarge"
}

variable "vpc_id" {
  description = "ID of the VPC where Neo4j will be deployed"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs where Neo4j can be deployed"
  type        = list(string)
}

variable "security_group_id" {
  description = "Security group ID for Neo4j instance"
  type        = string
}

variable "key_name" {
  description = "Name of the AWS key pair for SSH access"
  type        = string
  default     = null
}

variable "database_password" {
  description = "Password for the Neo4j database"
  type        = string
  sensitive   = true
}

variable "root_volume_size" {
  description = "Size of the root volume in GB"
  type        = number
  default     = 20
}

variable "data_volume_size" {
  description = "Size of the data volume in GB"
  type        = number
  default     = 100
}

variable "data_volume_iops" {
  description = "IOPS for the data volume"
  type        = number
  default     = 3000
}

variable "data_volume_throughput" {
  description = "Throughput for the data volume in MB/s"
  type        = number
  default     = 125
}

variable "heap_size" {
  description = "Neo4j heap size (e.g., 4G, 8G)"
  type        = string
  default     = "4G"
}

variable "pagecache_size" {
  description = "Neo4j page cache size (e.g., 2G, 4G)"
  type        = string
  default     = "2G"
}

variable "enable_deletion_protection" {
  description = "Enable deletion protection for the load balancer"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
}

variable "monitoring_enabled" {
  description = "Enable detailed monitoring"
  type        = bool
  default     = true
}

variable "tags" {
  description = "A map of tags to assign to the resource"
  type        = map(string)
  default     = {}
}