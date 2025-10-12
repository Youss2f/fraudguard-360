# ==============================================================================
# FraudGuard 360 - EKS Module Outputs
# Output values from EKS cluster creation
# ==============================================================================

output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = aws_eks_cluster.main.name
}

output "cluster_arn" {
  description = "The Amazon Resource Name (ARN) of the cluster"
  value       = aws_eks_cluster.main.arn
}

output "cluster_endpoint" {
  description = "Endpoint for your Kubernetes API server"
  value       = aws_eks_cluster.main.endpoint
}

output "cluster_version" {
  description = "The Kubernetes version for the cluster"
  value       = aws_eks_cluster.main.version
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = aws_eks_cluster.main.certificate_authority[0].data
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_security_group.cluster.id
}

output "node_security_group_id" {
  description = "Security group ID attached to the EKS node groups"
  value       = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN of the EKS cluster"
  value       = aws_iam_role.cluster.arn
}

output "node_group_iam_role_arn" {
  description = "IAM role ARN of the EKS node groups"
  value       = aws_iam_role.node_group.arn
}

output "oidc_provider_arn" {
  description = "The ARN of the OIDC Provider for the EKS cluster"
  value       = aws_iam_openid_connect_provider.cluster.arn
}

output "oidc_provider_url" {
  description = "The URL of the OIDC Provider for the EKS cluster"
  value       = aws_iam_openid_connect_provider.cluster.url
}

output "cluster_primary_security_group_id" {
  description = "The cluster primary security group ID created by EKS"
  value       = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
}

output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group for cluster logs"
  value       = aws_cloudwatch_log_group.cluster.name
}

output "cloudwatch_log_group_arn" {
  description = "ARN of the CloudWatch log group for cluster logs"  
  value       = aws_cloudwatch_log_group.cluster.arn
}

output "cluster_status" {
  description = "Status of the EKS cluster"
  value       = aws_eks_cluster.main.status
}

output "node_groups" {
  description = "Map of node groups created"
  value = {
    for k, v in aws_eks_node_group.main : k => {
      arn           = v.arn
      status        = v.status
      capacity_type = v.capacity_type
      instance_types = v.instance_types
      scaling_config = v.scaling_config
    }
  }
}

output "cluster_addons" {
  description = "Map of cluster addons"
  value = {
    for k, v in aws_eks_addon.main : k => {
      arn               = v.arn
      addon_version     = v.addon_version
      status            = v.status
      created_at        = v.created_at
      modified_at       = v.modified_at
    }
  }
}