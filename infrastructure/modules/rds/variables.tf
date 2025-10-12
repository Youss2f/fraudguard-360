# ==============================================================================
# FraudGuard 360 - RDS PostgreSQL Module Variables
# Input variables for the RDS PostgreSQL module
# ==============================================================================

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "db_identifier" {
  description = "Identifier for the RDS instance"
  type        = string
  default     = "fraudguard-postgres"
}

variable "vpc_id" {
  description = "VPC ID where RDS will be deployed"
  type        = string
}

variable "database_subnet_ids" {
  description = "List of database subnet IDs"
  type        = list(string)
}

variable "allowed_cidr_blocks" {
  description = "List of CIDR blocks allowed to connect to RDS"
  type        = list(string)
  default     = ["10.0.0.0/8"]
}

variable "common_tags" {
  description = "Common tags to be applied to all resources"
  type        = map(string)
  default     = {}
}

# Database Configuration
variable "engine_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "15.4"
}

variable "instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
  
  validation {
    condition = can(regex("^db\\.", var.instance_class))
    error_message = "Instance class must be a valid RDS instance type (e.g., db.t3.medium)."
  }
}

variable "allocated_storage" {
  description = "Initial allocated storage in GB"
  type        = number
  default     = 20
  
  validation {
    condition     = var.allocated_storage >= 20
    error_message = "Allocated storage must be at least 20 GB."
  }
}

variable "max_allocated_storage" {
  description = "Maximum allocated storage for autoscaling in GB"
  type        = number
  default     = 1000
  
  validation {
    condition     = var.max_allocated_storage >= var.allocated_storage
    error_message = "Max allocated storage must be greater than or equal to allocated storage."
  }
}

variable "storage_type" {
  description = "Storage type for RDS instance"
  type        = string
  default     = "gp3"
  
  validation {
    condition     = contains(["gp2", "gp3", "io1", "io2"], var.storage_type)
    error_message = "Storage type must be one of: gp2, gp3, io1, io2."
  }
}

variable "database_name" {
  description = "Name of the initial database"
  type        = string
  default     = "fraudguard"
}

variable "master_username" {
  description = "Master username for the database"
  type        = string
  default     = "postgres"
}

# Backup Configuration
variable "backup_window" {
  description = "Preferred backup window"
  type        = string
  default     = "03:00-04:00"
}

variable "backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
  default     = 7
  
  validation {
    condition     = var.backup_retention_period >= 0 && var.backup_retention_period <= 35
    error_message = "Backup retention period must be between 0 and 35 days."
  }
}

variable "maintenance_window" {
  description = "Preferred maintenance window"
  type        = string
  default     = "Mon:04:00-Mon:05:00"
}

variable "auto_minor_version_upgrade" {
  description = "Enable auto minor version upgrade"
  type        = bool
  default     = true
}

variable "skip_final_snapshot" {
  description = "Skip final snapshot when destroying"
  type        = bool
  default     = false
}

variable "deletion_protection" {
  description = "Enable deletion protection"
  type        = bool
  default     = true
}

variable "deletion_window_in_days" {
  description = "KMS key deletion window in days"
  type        = number
  default     = 7
  
  validation {
    condition     = var.deletion_window_in_days >= 7 && var.deletion_window_in_days <= 30
    error_message = "KMS key deletion window must be between 7 and 30 days."
  }
}

# Monitoring Configuration
variable "monitoring_interval" {
  description = "Enhanced monitoring interval in seconds (0, 1, 5, 10, 15, 30, 60)"
  type        = number
  default     = 60
  
  validation {
    condition     = contains([0, 1, 5, 10, 15, 30, 60], var.monitoring_interval)
    error_message = "Monitoring interval must be one of: 0, 1, 5, 10, 15, 30, 60."
  }
}

variable "enabled_cloudwatch_logs_exports" {
  description = "List of log types to export to CloudWatch"
  type        = list(string)
  default     = ["postgresql", "upgrade"]
  
  validation {
    condition = alltrue([
      for log_type in var.enabled_cloudwatch_logs_exports :
      contains(["postgresql", "upgrade"], log_type)
    ])
    error_message = "Log types must be from: postgresql, upgrade."
  }
}

variable "cloudwatch_log_retention_days" {
  description = "CloudWatch log retention period in days"
  type        = number
  default     = 30
  
  validation {
    condition     = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.cloudwatch_log_retention_days)
    error_message = "Log retention must be a valid CloudWatch retention period."
  }
}

variable "performance_insights_enabled" {
  description = "Enable Performance Insights"
  type        = bool
  default     = true
}

variable "performance_insights_retention_period" {
  description = "Performance Insights retention period in days"
  type        = number
  default     = 7
  
  validation {
    condition     = contains([7, 731], var.performance_insights_retention_period)
    error_message = "Performance Insights retention must be 7 or 731 days."
  }
}

# Read Replica Configuration
variable "create_read_replica" {
  description = "Create a read replica"
  type        = bool
  default     = false
}

variable "read_replica_instance_class" {
  description = "Instance class for read replica"
  type        = string
  default     = "db.t3.medium"
}

# Database Parameters
variable "db_parameters" {
  description = "Database parameters to set"
  type        = map(string)
  default = {
    "shared_preload_libraries" = "pg_stat_statements"
    "log_statement"            = "all"
    "log_min_duration_statement" = "1000"
    "log_checkpoints"          = "on"
    "log_connections"          = "on"
    "log_disconnections"       = "on"
    "log_lock_waits"          = "on"
    "log_temp_files"          = "0"
    "track_activity_query_size" = "2048"
    "pg_stat_statements.track" = "all"
  }
}

# Alerting Configuration
variable "alarm_actions" {
  description = "List of ARNs for alarm actions (SNS topics)"
  type        = list(string)
  default     = []
}

variable "sns_topic_arn" {
  description = "SNS topic ARN for RDS events"
  type        = string
  default     = ""
}

variable "event_categories" {
  description = "List of event categories for RDS event subscription"
  type        = list(string)
  default     = ["availability", "backup", "configuration change", "creation", "deletion", "failover", "failure", "low storage", "maintenance", "notification", "read replica", "recovery", "restoration"]
}