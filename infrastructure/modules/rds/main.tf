# ==============================================================================
# FraudGuard 360 - RDS PostgreSQL Module
# Managed PostgreSQL database for application data
# ==============================================================================

resource "aws_db_subnet_group" "postgresql" {
  name       = "${var.environment}-${var.db_identifier}-subnet-group"
  subnet_ids = var.database_subnet_ids

  tags = merge(var.common_tags, {
    Name = "${var.environment}-${var.db_identifier}-subnet-group"
    Type = "Database"
  })
}

resource "aws_db_parameter_group" "postgresql" {
  family = "postgres${split(".", var.engine_version)[0]}"
  name   = "${var.environment}-${var.db_identifier}-params"

  dynamic "parameter" {
    for_each = var.db_parameters
    content {
      name  = parameter.key
      value = parameter.value
    }
  }

  tags = merge(var.common_tags, {
    Name = "${var.environment}-${var.db_identifier}-params"
    Type = "Database"
  })
}

resource "aws_security_group" "rds" {
  name        = "${var.environment}-${var.db_identifier}-rds-sg"
  description = "Security group for RDS PostgreSQL instance"
  vpc_id      = var.vpc_id

  ingress {
    description = "PostgreSQL from application"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.common_tags, {
    Name = "${var.environment}-${var.db_identifier}-rds-sg"
    Type = "Database"
  })
}

# KMS key for RDS encryption
resource "aws_kms_key" "rds" {
  description             = "KMS key for RDS encryption in ${var.environment}"
  deletion_window_in_days = var.deletion_window_in_days
  enable_key_rotation     = true

  tags = merge(var.common_tags, {
    Name = "${var.environment}-${var.db_identifier}-kms-key"
    Type = "Encryption"
  })
}

resource "aws_kms_alias" "rds" {
  name          = "alias/${var.environment}-${var.db_identifier}-rds"
  target_key_id = aws_kms_key.rds.key_id
}

# Random password for master user
resource "random_password" "master" {
  length  = 32
  special = true
}

# Store master password in AWS Secrets Manager
resource "aws_secretsmanager_secret" "rds_master" {
  name                    = "${var.environment}-${var.db_identifier}-master-password"
  description             = "Master password for RDS PostgreSQL instance"
  recovery_window_in_days = var.deletion_window_in_days
  kms_key_id              = aws_kms_key.rds.arn

  tags = merge(var.common_tags, {
    Name = "${var.environment}-${var.db_identifier}-master-password"
    Type = "Secret"
  })
}

resource "aws_secretsmanager_secret_version" "rds_master" {
  secret_id = aws_secretsmanager_secret.rds_master.id
  secret_string = jsonencode({
    username = var.master_username
    password = random_password.master.result
    engine   = "postgres"
    host     = aws_db_instance.postgresql.endpoint
    port     = aws_db_instance.postgresql.port
    dbname   = aws_db_instance.postgresql.db_name
  })
}

# Main RDS instance
resource "aws_db_instance" "postgresql" {
  identifier     = "${var.environment}-${var.db_identifier}"
  engine         = "postgres"
  engine_version = var.engine_version
  instance_class = var.instance_class

  allocated_storage     = var.allocated_storage
  max_allocated_storage = var.max_allocated_storage
  storage_type          = var.storage_type
  storage_encrypted     = true
  kms_key_id            = aws_kms_key.rds.arn

  db_name  = var.database_name
  username = var.master_username
  password = random_password.master.result

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.postgresql.name
  parameter_group_name   = aws_db_parameter_group.postgresql.name

  # Backup configuration
  backup_window      = var.backup_window
  backup_retention_period = var.backup_retention_period
  copy_tags_to_snapshot   = true
  delete_automated_backups = false

  # Maintenance
  maintenance_window         = var.maintenance_window
  auto_minor_version_upgrade = var.auto_minor_version_upgrade
  apply_immediately          = false

  # Monitoring
  monitoring_interval = var.monitoring_interval
  monitoring_role_arn = var.monitoring_interval > 0 ? aws_iam_role.rds_enhanced_monitoring[0].arn : null
  enabled_cloudwatch_logs_exports = var.enabled_cloudwatch_logs_exports

  # Performance Insights
  performance_insights_enabled          = var.performance_insights_enabled
  performance_insights_kms_key_id       = var.performance_insights_enabled ? aws_kms_key.rds.arn : null
  performance_insights_retention_period = var.performance_insights_retention_period

  # Security
  publicly_accessible = false
  skip_final_snapshot = var.skip_final_snapshot
  final_snapshot_identifier = var.skip_final_snapshot ? null : "${var.environment}-${var.db_identifier}-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"

  # Deletion protection
  deletion_protection = var.deletion_protection

  depends_on = [
    aws_cloudwatch_log_group.postgresql
  ]

  tags = merge(var.common_tags, {
    Name = "${var.environment}-${var.db_identifier}"
    Type = "Database"
  })
}

# Read replica (optional)
resource "aws_db_instance" "postgresql_read_replica" {
  count = var.create_read_replica ? 1 : 0

  identifier             = "${var.environment}-${var.db_identifier}-read-replica"
  replicate_source_db    = aws_db_instance.postgresql.id
  instance_class         = var.read_replica_instance_class
  publicly_accessible    = false
  auto_minor_version_upgrade = var.auto_minor_version_upgrade

  # Monitoring
  monitoring_interval = var.monitoring_interval
  monitoring_role_arn = var.monitoring_interval > 0 ? aws_iam_role.rds_enhanced_monitoring[0].arn : null

  # Performance Insights
  performance_insights_enabled          = var.performance_insights_enabled
  performance_insights_kms_key_id       = var.performance_insights_enabled ? aws_kms_key.rds.arn : null
  performance_insights_retention_period = var.performance_insights_retention_period

  tags = merge(var.common_tags, {
    Name = "${var.environment}-${var.db_identifier}-read-replica"
    Type = "Database"
  })
}

# Enhanced monitoring IAM role
resource "aws_iam_role" "rds_enhanced_monitoring" {
  count = var.monitoring_interval > 0 ? 1 : 0

  name = "${var.environment}-${var.db_identifier}-enhanced-monitoring"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(var.common_tags, {
    Name = "${var.environment}-${var.db_identifier}-enhanced-monitoring"
    Type = "IAM"
  })
}

resource "aws_iam_role_policy_attachment" "rds_enhanced_monitoring" {
  count = var.monitoring_interval > 0 ? 1 : 0

  role       = aws_iam_role.rds_enhanced_monitoring[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "postgresql" {
  for_each = toset(var.enabled_cloudwatch_logs_exports)

  name              = "/aws/rds/instance/${var.environment}-${var.db_identifier}/${each.value}"
  retention_in_days = var.cloudwatch_log_retention_days
  kms_key_id        = aws_kms_key.rds.arn

  tags = merge(var.common_tags, {
    Name = "${var.environment}-${var.db_identifier}-${each.value}-logs"
    Type = "Logging"
  })
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "database_cpu" {
  alarm_name          = "${var.environment}-${var.db_identifier}-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = "120"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors RDS CPU utilization"
  alarm_actions       = var.alarm_actions

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.postgresql.id
  }

  tags = merge(var.common_tags, {
    Name = "${var.environment}-${var.db_identifier}-high-cpu"
    Type = "Alarm"
  })
}

resource "aws_cloudwatch_metric_alarm" "database_connections" {
  alarm_name          = "${var.environment}-${var.db_identifier}-high-connections"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "DatabaseConnections"
  namespace           = "AWS/RDS"
  period              = "120"
  statistic           = "Average"
  threshold           = "80" # Adjust based on your instance class
  alarm_description   = "This metric monitors RDS connection count"
  alarm_actions       = var.alarm_actions

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.postgresql.id
  }

  tags = merge(var.common_tags, {
    Name = "${var.environment}-${var.db_identifier}-high-connections"
    Type = "Alarm"
  })
}

resource "aws_cloudwatch_metric_alarm" "database_free_storage_space" {
  alarm_name          = "${var.environment}-${var.db_identifier}-low-storage"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "FreeStorageSpace"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "2000000000" # 2GB in bytes
  alarm_description   = "This metric monitors RDS free storage space"
  alarm_actions       = var.alarm_actions

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.postgresql.id
  }

  tags = merge(var.common_tags, {
    Name = "${var.environment}-${var.db_identifier}-low-storage"
    Type = "Alarm"
  })
}

# DB Event Subscription
resource "aws_db_event_subscription" "postgresql" {
  count = length(var.event_categories) > 0 ? 1 : 0

  name      = "${var.environment}-${var.db_identifier}-events"
  sns_topic = var.sns_topic_arn

  source_type = "db-instance"
  source_ids  = [aws_db_instance.postgresql.id]

  event_categories = var.event_categories

  tags = merge(var.common_tags, {
    Name = "${var.environment}-${var.db_identifier}-events"
    Type = "Notification"
  })
}