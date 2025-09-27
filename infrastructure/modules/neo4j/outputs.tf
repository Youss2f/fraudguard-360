# ==============================================================================
# FraudGuard 360 - Neo4j Module Outputs
# Output values from Neo4j deployment
# ==============================================================================

output "instance_name" {
  description = "Name of the Neo4j instance"
  value       = var.instance_name
}

output "load_balancer_dns_name" {
  description = "DNS name of the Neo4j load balancer"
  value       = aws_lb.neo4j.dns_name
}

output "load_balancer_arn" {
  description = "ARN of the Neo4j load balancer"
  value       = aws_lb.neo4j.arn
}

output "endpoint" {
  description = "Neo4j endpoint for connections"
  value       = aws_lb.neo4j.dns_name
}

output "bolt_uri" {
  description = "Neo4j Bolt URI for connections"
  value       = "bolt://${aws_lb.neo4j.dns_name}:7687"
}

output "http_uri" {
  description = "Neo4j HTTP URI for browser access"
  value       = "http://${aws_lb.neo4j.dns_name}:7474"
}

output "https_uri" {
  description = "Neo4j HTTPS URI for secure browser access"
  value       = "https://${aws_lb.neo4j.dns_name}:7473"
}

output "autoscaling_group_arn" {
  description = "ARN of the Auto Scaling Group"
  value       = aws_autoscaling_group.neo4j.arn
}

output "autoscaling_group_name" {
  description = "Name of the Auto Scaling Group"
  value       = aws_autoscaling_group.neo4j.name
}

output "launch_template_id" {
  description = "ID of the launch template"
  value       = aws_launch_template.neo4j.id
}

output "launch_template_version" {
  description = "Latest version of the launch template"
  value       = aws_launch_template.neo4j.latest_version
}

output "iam_role_arn" {
  description = "ARN of the IAM role"
  value       = aws_iam_role.neo4j.arn
}

output "iam_role_name" {
  description = "Name of the IAM role"
  value       = aws_iam_role.neo4j.name
}

output "target_group_bolt_arn" {
  description = "ARN of the Bolt target group"
  value       = aws_lb_target_group.neo4j_bolt.arn
}

output "target_group_http_arn" {
  description = "ARN of the HTTP target group"
  value       = aws_lb_target_group.neo4j_http.arn
}

output "connection_info" {
  description = "Connection information for applications"
  value = {
    bolt_uri     = "bolt://${aws_lb.neo4j.dns_name}:7687"
    http_uri     = "http://${aws_lb.neo4j.dns_name}:7474"
    https_uri    = "https://${aws_lb.neo4j.dns_name}:7473"
    endpoint     = aws_lb.neo4j.dns_name
    bolt_port    = 7687
    http_port    = 7474
    https_port   = 7473
  }
  sensitive = true
}