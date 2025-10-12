# ==============================================================================
# FraudGuard 360 - Kafka Module Outputs
# Output values from MSK cluster creation
# ==============================================================================

output "cluster_arn" {
  description = "Amazon Resource Name (ARN) of the MSK cluster"
  value       = aws_msk_cluster.main.arn
}

output "cluster_name" {
  description = "MSK cluster name"
  value       = aws_msk_cluster.main.cluster_name
}

output "bootstrap_brokers" {
  description = "Comma separated list of one or more hostname:port pairs of kafka brokers suitable to bootstrap connectivity to the kafka cluster"
  value       = aws_msk_cluster.main.bootstrap_brokers
}

output "bootstrap_brokers_tls" {
  description = "Comma separated list of one or more DNS names (or IP addresses) and TLS port pairs kafka brokers suitable to bootstrap connectivity to the kafka cluster"
  value       = aws_msk_cluster.main.bootstrap_brokers_tls
}

output "bootstrap_brokers_sasl_scram" {
  description = "Comma separated list of one or more DNS names (or IP addresses) and SASL SCRAM port pairs kafka brokers suitable to bootstrap connectivity to the kafka cluster"
  value       = aws_msk_cluster.main.bootstrap_brokers_sasl_scram
}

output "zookeeper_connect_string" {
  description = "A comma separated list of one or more hostname:port pairs to use to connect to the Apache Zookeeper cluster"
  value       = aws_msk_cluster.main.zookeeper_connect_string
}

output "current_version" {
  description = "Current version of the MSK Cluster used for updates"
  value       = aws_msk_cluster.main.current_version
}

output "security_group_id" {
  description = "Security group ID for the MSK cluster"
  value       = aws_security_group.msk.id
}

output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group for MSK logs"
  value       = var.cloudwatch_logs_enabled ? aws_cloudwatch_log_group.msk[0].name : null
}

output "cloudwatch_log_group_arn" {
  description = "ARN of the CloudWatch log group for MSK logs"
  value       = var.cloudwatch_logs_enabled ? aws_cloudwatch_log_group.msk[0].arn : null
}

output "kafka_topics" {
  description = "List of Kafka topics configured"
  value = [
    for topic in var.kafka_topics : {
      name               = topic.name
      partitions         = topic.partitions
      replication_factor = topic.replication_factor
    }
  ]
}