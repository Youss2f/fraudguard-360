# Terraform and Provider Version Constraints
# FraudGuard-360 Infrastructure

terraform {
  required_version = ">= 1.5.0, < 2.0.0"

  required_providers {
    oci = {
      source  = "oracle/oci"
      version = "~> 5.46.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0.0"
    }
  }

  # Recommended: Configure remote backend for team collaboration
  # backend "s3" {
  #   bucket         = "fraudguard-terraform-state"
  #   key            = "infrastructure/terraform.tfstate"
  #   region         = "us-ashburn-1"
  #   encrypt        = true
  #   dynamodb_table = "terraform-state-lock"
  # }
}
