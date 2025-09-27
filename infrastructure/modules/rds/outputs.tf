# ==============================================================================
# FraudGuard 360 - RDS PostgreSQL Module Outputs
# Output values from RDS deployment
# ==============================================================================

output "db_instance_id" {
  description = "RDS instance identifier"
  value       = aws_db_instance.postgresql.id
}

output "db_instance_arn" {
  description = "RDS instance ARN"
  value       = aws_db_instance.postgresql.arn
}

output "db_instance_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.postgresql.endpoint
}

output "db_instance_port" {
  description = "RDS instance port"
  value       = aws_db_instance.postgresql.port
}

output "db_instance_address" {
  description = "RDS instance hostname"
  value       = aws_db_instance.postgresql.address
}

output "db_instance_availability_zone" {
  description = "RDS instance availability zone"
  value       = aws_db_instance.postgresql.availability_zone
}

output "database_name" {
  description = "Database name"
  value       = aws_db_instance.postgresql.db_name
}

output "master_username" {
  description = "Master username"
  value       = aws_db_instance.postgresql.username
  sensitive   = true
}

output "secrets_manager_secret_arn" {
  description = "ARN of the Secrets Manager secret containing database credentials"
  value       = aws_secretsmanager_secret.rds_master.arn
}

output "secrets_manager_secret_name" {
  description = "Name of the Secrets Manager secret containing database credentials"
  value       = aws_secretsmanager_secret.rds_master.name
}

output "kms_key_id" {
  description = "KMS key ID used for encryption"
  value       = aws_kms_key.rds.key_id
}

output "kms_key_arn" {
  description = "KMS key ARN used for encryption"
  value       = aws_kms_key.rds.arn
}

output "db_subnet_group_id" {
  description = "DB subnet group identifier"
  value       = aws_db_subnet_group.postgresql.id
}

output "db_subnet_group_arn" {
  description = "DB subnet group ARN"
  value       = aws_db_subnet_group.postgresql.arn
}

output "db_parameter_group_id" {
  description = "DB parameter group identifier"
  value       = aws_db_parameter_group.postgresql.id
}

output "db_parameter_group_arn" {
  description = "DB parameter group ARN"
  value       = aws_db_parameter_group.postgresql.arn
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.rds.id
}

output "security_group_arn" {
  description = "Security group ARN"
  value       = aws_security_group.rds.arn
}

output "enhanced_monitoring_iam_role_arn" {
  description = "Enhanced monitoring IAM role ARN"
  value       = var.monitoring_interval > 0 ? aws_iam_role.rds_enhanced_monitoring[0].arn : null
}

output "read_replica_endpoint" {
  description = "Read replica endpoint"
  value       = var.create_read_replica ? aws_db_instance.postgresql_read_replica[0].endpoint : null
}

output "read_replica_port" {
  description = "Read replica port"
  value       = var.create_read_replica ? aws_db_instance.postgresql_read_replica[0].port : null
}

output "read_replica_address" {
  description = "Read replica hostname"
  value       = var.create_read_replica ? aws_db_instance.postgresql_read_replica[0].address : null
}

output "connection_info" {
  description = "Database connection information for applications"
  value = {
    endpoint     = aws_db_instance.postgresql.endpoint
    port         = aws_db_instance.postgresql.port
    database     = aws_db_instance.postgresql.db_name
    username     = aws_db_instance.postgresql.username
    secret_arn   = aws_secretsmanager_secret.rds_master.arn
    secret_name  = aws_secretsmanager_secret.rds_master.name
    read_replica_endpoint = var.create_read_replica ? aws_db_instance.postgresql_read_replica[0].endpoint : null
  }
  sensitive = true
}

output "cloudwatch_log_groups" {
  description = "CloudWatch log group names"
  value = {
    for log_type in var.enabled_cloudwatch_logs_exports :
    log_type => aws_cloudwatch_log_group.postgresql[log_type].name
  }
}

output "backup_info" {
  description = "Backup configuration information"
  value = {
    backup_window           = aws_db_instance.postgresql.backup_window
    backup_retention_period = aws_db_instance.postgresql.backup_retention_period
    maintenance_window      = aws_db_instance.postgresql.maintenance_window
  }
}