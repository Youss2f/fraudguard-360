# ==============================================================================
# FraudGuard 360 - Kafka Module Variables
# Input variables for MSK cluster configuration
# ==============================================================================

variable "cluster_name" {
  description = "Name of the MSK cluster"
  type        = string
}

variable "kafka_version" {
  description = "Kafka version for the MSK cluster"
  type        = string
  default     = "2.8.1"
}

variable "vpc_id" {
  description = "ID of the VPC where the MSK cluster will be created"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs where the MSK cluster will be created"
  type        = list(string)
}

variable "broker_node_instance_type" {
  description = "Instance type for MSK broker nodes"
  type        = string
  default     = "kafka.m5.large"
}

variable "broker_node_storage_size" {
  description = "Storage size per broker node in GB"
  type        = number
  default     = 100
}

variable "allowed_cidr_blocks" {
  description = "List of CIDR blocks allowed to access the MSK cluster"
  type        = list(string)
  default     = ["10.0.0.0/8"]
}

variable "encryption_in_transit_client_broker" {
  description = "Encryption setting for data in transit between clients and brokers"
  type        = string
  default     = "TLS"
  validation {
    condition     = contains(["TLS", "TLS_PLAINTEXT", "PLAINTEXT"], var.encryption_in_transit_client_broker)
    error_message = "Valid values are TLS, TLS_PLAINTEXT, or PLAINTEXT."
  }
}

variable "encryption_in_transit_in_cluster" {
  description = "Whether data communication among broker nodes is encrypted"
  type        = bool
  default     = true
}

variable "encryption_at_rest_kms_key_id" {
  description = "KMS key ID for encryption at rest"
  type        = string
  default     = null
}

variable "enhanced_monitoring" {
  description = "Enhanced monitoring level"
  type        = string
  default     = "DEFAULT"
  validation {
    condition     = contains(["DEFAULT", "PER_BROKER", "PER_TOPIC_PER_BROKER", "PER_TOPIC_PER_PARTITION"], var.enhanced_monitoring)
    error_message = "Valid values are DEFAULT, PER_BROKER, PER_TOPIC_PER_BROKER, or PER_TOPIC_PER_PARTITION."
  }
}

variable "cloudwatch_logs_enabled" {
  description = "Enable CloudWatch logs"
  type        = bool
  default     = true
}

variable "cloudwatch_logs_retention_days" {
  description = "CloudWatch logs retention period in days"
  type        = number
  default     = 30
}

variable "firehose_logs_enabled" {
  description = "Enable Kinesis Data Firehose logs"
  type        = bool
  default     = false
}

variable "firehose_delivery_stream" {
  description = "Name of the Kinesis Data Firehose delivery stream"
  type        = string
  default     = null
}

variable "s3_logs_enabled" {
  description = "Enable S3 logs"
  type        = bool
  default     = false
}

variable "s3_logs_bucket" {
  description = "Name of the S3 bucket for logs"
  type        = string
  default     = null
}

variable "s3_logs_prefix" {
  description = "Prefix for S3 logs"
  type        = string
  default     = "kafka-logs/"
}

variable "kafka_topics" {
  description = "List of Kafka topics to create"
  type = list(object({
    name               = string
    partitions         = number
    replication_factor = number
  }))
  default = [
    {
      name               = "fraud-transactions"
      partitions         = 6
      replication_factor = 2
    },
    {
      name               = "fraud-alerts"
      partitions         = 3
      replication_factor = 2
    },
    {
      name               = "fraud-analytics"
      partitions         = 3
      replication_factor = 2
    }
  ]
}

variable "tags" {
  description = "A map of tags to assign to the resource"
  type        = map(string)
  default     = {}
}