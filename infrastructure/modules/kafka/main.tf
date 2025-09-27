# ==============================================================================
# FraudGuard 360 - MSK (Managed Streaming for Kafka) Module
# Fully managed Apache Kafka service with security and monitoring
# ==============================================================================

# ==============================================================================
# MSK Cluster
# ==============================================================================

resource "aws_msk_cluster" "main" {
  cluster_name           = var.cluster_name
  kafka_version          = var.kafka_version
  number_of_broker_nodes = length(var.subnet_ids)

  broker_node_group_info {
    instance_type   = var.broker_node_instance_type
    client_subnets  = var.subnet_ids
    storage_info {
      ebs_storage_info {
        volume_size = var.broker_node_storage_size
      }
    }
    security_groups = [aws_security_group.msk.id]
  }

  # Kafka Configuration
  configuration_info {
    arn      = aws_msk_configuration.main.arn
    revision = aws_msk_configuration.main.latest_revision
  }

  # Encryption in transit
  encryption_info {
    encryption_in_transit {
      client_broker = var.encryption_in_transit_client_broker
      in_cluster    = var.encryption_in_transit_in_cluster
    }
    encryption_at_rest_kms_key_id = var.encryption_at_rest_kms_key_id
  }

  # Enhanced monitoring
  enhanced_monitoring = var.enhanced_monitoring

  # Logging configuration
  logging_info {
    broker_logs {
      dynamic "cloudwatch_logs" {
        for_each = var.cloudwatch_logs_enabled ? [1] : []
        content {
          enabled   = true
          log_group = aws_cloudwatch_log_group.msk[0].name
        }
      }

      dynamic "firehose" {
        for_each = var.firehose_logs_enabled ? [1] : []
        content {
          enabled         = true
          delivery_stream = var.firehose_delivery_stream
        }
      }

      dynamic "s3" {
        for_each = var.s3_logs_enabled ? [1] : []
        content {
          enabled = true
          bucket  = var.s3_logs_bucket
          prefix  = var.s3_logs_prefix
        }
      }
    }
  }

  tags = var.tags
}

# ==============================================================================
# MSK Configuration
# ==============================================================================

resource "aws_msk_configuration" "main" {
  name           = "${var.cluster_name}-config"
  description    = "Configuration for ${var.cluster_name} MSK cluster"
  kafka_versions = [var.kafka_version]

  server_properties = <<-EOT
    auto.create.topics.enable=false
    default.replication.factor=2
    min.insync.replicas=2
    num.partitions=6
    num.replica.fetchers=2
    replica.lag.time.max.ms=30000
    socket.receive.buffer.bytes=102400
    socket.request.max.bytes=104857600
    socket.send.buffer.bytes=102400
    unclean.leader.election.enable=false
    zookeeper.session.timeout.ms=18000
    transaction.state.log.replication.factor=2
    transaction.state.log.min.isr=2
    log.retention.hours=168
    log.segment.bytes=1073741824
    log.retention.check.interval.ms=300000
    log.cleanup.policy=delete
  EOT
}

# ==============================================================================
# Security Group for MSK
# ==============================================================================

resource "aws_security_group" "msk" {
  name        = "${var.cluster_name}-msk-sg"
  description = "Security group for MSK cluster"
  vpc_id      = var.vpc_id

  # Kafka plaintext
  ingress {
    from_port   = 9092
    to_port     = 9092
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  # Kafka TLS
  ingress {
    from_port   = 9094
    to_port     = 9094
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  # Kafka SASL/SCRAM
  ingress {
    from_port   = 9096
    to_port     = 9096
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  # Zookeeper
  ingress {
    from_port   = 2181
    to_port     = 2181
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  # JMX monitoring
  ingress {
    from_port   = 11001
    to_port     = 11002
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, {
    Name = "${var.cluster_name}-msk-sg"
  })
}

# ==============================================================================
# CloudWatch Log Group (Optional)
# ==============================================================================

resource "aws_cloudwatch_log_group" "msk" {
  count             = var.cloudwatch_logs_enabled ? 1 : 0
  name              = "/aws/msk/${var.cluster_name}"
  retention_in_days = var.cloudwatch_logs_retention_days

  tags = var.tags
}

# ==============================================================================
# Create Kafka Topics (using null_resource with kafka-topics.sh)
# ==============================================================================

resource "null_resource" "kafka_topics" {
  count = length(var.kafka_topics)

  triggers = {
    bootstrap_brokers = aws_msk_cluster.main.bootstrap_brokers_tls
    topic_name       = var.kafka_topics[count.index].name
    partitions       = var.kafka_topics[count.index].partitions
    replication      = var.kafka_topics[count.index].replication_factor
  }

  # This would typically run from a bastion host or CI/CD pipeline
  # with access to the MSK cluster
  provisioner "local-exec" {
    command = <<-EOT
      echo "Topic creation should be handled by application deployment or separate automation"
      echo "Topic: ${var.kafka_topics[count.index].name}"
      echo "Partitions: ${var.kafka_topics[count.index].partitions}"
      echo "Replication Factor: ${var.kafka_topics[count.index].replication_factor}"
    EOT
  }

  depends_on = [aws_msk_cluster.main]
}
  chart      = "kafka"
  namespace  = "fraudguard"

  set = [
    {
      name  = "replicaCount"
      value = "3"
    }
  ]
  # Additional configs for partitions, replication
}
