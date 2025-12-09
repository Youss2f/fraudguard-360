# Input Variables for FraudGuard-360 Infrastructure

variable "tenancy_ocid" {
  description = "The OCID of the tenancy"
  type        = string
}

variable "user_ocid" {
  description = "The OCID of the user"
  type        = string
}

variable "fingerprint" {
  description = "The fingerprint of the API key"
  type        = string
}

variable "private_key_path" {
  description = "The path to the private key file"
  type        = string
}

variable "region" {
  description = "The OCI region"
  type        = string
  default     = "us-ashburn-1"
}

variable "compartment_ocid" {
  description = "The OCID of the compartment"
  type        = string
}

variable "project_name" {
  description = "Name of the project (used for resource naming)"
  type        = string
  default     = "fraudguard"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "production"
}

variable "vcn_cidr" {
  description = "CIDR block for the VCN"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidr" {
  description = "CIDR block for the public subnet"
  type        = string
  default     = "10.0.1.0/24"
}

variable "private_subnet_cidr" {
  description = "CIDR block for the private subnet"
  type        = string
  default     = "10.0.2.0/24"
}

variable "allowed_k8s_cidr" {
  description = "CIDR block allowed to access Kubernetes API"
  type        = string
  default     = "0.0.0.0/0"
}

variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "FraudGuard-360"
    ManagedBy   = "Terraform"
    Environment = "production"
    Owner       = "DevOps Team"
  }
}

# =============================================================================
# Compute Instance Configuration
# =============================================================================

variable "node_pool_size" {
  description = "Number of nodes in the Kubernetes node pool"
  type        = number
  default     = 3

  validation {
    condition     = var.node_pool_size >= 1 && var.node_pool_size <= 10
    error_message = "Node pool size must be between 1 and 10."
  }
}

variable "node_shape" {
  description = "Shape (instance type) for Kubernetes worker nodes"
  type        = string
  default     = "VM.Standard.E4.Flex"

  validation {
    condition = contains([
      "VM.Standard.E4.Flex",
      "VM.Standard.E3.Flex",
      "VM.Standard.A1.Flex",
      "VM.Standard3.Flex"
    ], var.node_shape)
    error_message = "Node shape must be a valid OCI flexible shape."
  }
}

variable "node_ocpus" {
  description = "Number of OCPUs for each worker node (for Flex shapes)"
  type        = number
  default     = 2

  validation {
    condition     = var.node_ocpus >= 1 && var.node_ocpus <= 64
    error_message = "OCPUs must be between 1 and 64."
  }
}

variable "node_memory_gb" {
  description = "Memory in GB for each worker node (for Flex shapes)"
  type        = number
  default     = 16

  validation {
    condition     = var.node_memory_gb >= 1 && var.node_memory_gb <= 1024
    error_message = "Memory must be between 1 and 1024 GB."
  }
}

variable "kubernetes_version" {
  description = "Kubernetes version for OKE cluster"
  type        = string
  default     = "v1.28.2"
}

variable "boot_volume_size_gb" {
  description = "Boot volume size in GB for worker nodes"
  type        = number
  default     = 50

  validation {
    condition     = var.boot_volume_size_gb >= 50 && var.boot_volume_size_gb <= 500
    error_message = "Boot volume must be between 50 and 500 GB."
  }
}

# =============================================================================
# Database Configuration
# =============================================================================

variable "db_system_shape" {
  description = "Shape for PostgreSQL database system"
  type        = string
  default     = "VM.Standard.E4.Flex"
}

variable "db_storage_gb" {
  description = "Storage size in GB for database"
  type        = number
  default     = 256

  validation {
    condition     = var.db_storage_gb >= 50 && var.db_storage_gb <= 65536
    error_message = "Database storage must be between 50 and 65536 GB."
  }
}
