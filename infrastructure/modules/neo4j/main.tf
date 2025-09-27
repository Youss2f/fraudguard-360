# ==============================================================================
# FraudGuard 360 - Neo4j Module
# High-performance graph database on EC2 with EBS optimization
# ==============================================================================

# ==============================================================================
# Data Sources
# ==============================================================================

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ==============================================================================
# Launch Template for Neo4j
# ==============================================================================

resource "aws_launch_template" "neo4j" {
  name_prefix   = "${var.instance_name}-"
  image_id      = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  key_name      = var.key_name

  vpc_security_group_ids = [var.security_group_id]

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    neo4j_password = var.database_password
    heap_size      = var.heap_size
    pagecache_size = var.pagecache_size
  }))

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size           = var.root_volume_size
      volume_type          = "gp3"
      encrypted            = true
      delete_on_termination = true
    }
  }

  block_device_mappings {
    device_name = "/dev/sdf"
    ebs {
      volume_size           = var.data_volume_size
      volume_type          = "gp3"
      iops                 = var.data_volume_iops
      throughput           = var.data_volume_throughput
      encrypted            = true
      delete_on_termination = false
    }
  }

  iam_instance_profile {
    name = aws_iam_instance_profile.neo4j.name
  }

  monitoring {
    enabled = true
  }

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
    http_put_response_hop_limit = 1
  }

  tag_specifications {
    resource_type = "instance"
    tags = merge(var.tags, {
      Name = var.instance_name
      Type = "Neo4j Database"
    })
  }

  tag_specifications {
    resource_type = "volume"
    tags = merge(var.tags, {
      Name = "${var.instance_name}-volume"
      Type = "Neo4j Storage"
    })
  }
}

# ==============================================================================
# Auto Scaling Group
# ==============================================================================

resource "aws_autoscaling_group" "neo4j" {
  name                = "${var.instance_name}-asg"
  vpc_zone_identifier = var.subnet_ids
  target_group_arns   = [aws_lb_target_group.neo4j_bolt.arn, aws_lb_target_group.neo4j_http.arn]
  health_check_type   = "ELB"
  health_check_grace_period = 300

  min_size         = 1
  max_size         = 1
  desired_capacity = 1

  launch_template {
    id      = aws_launch_template.neo4j.id
    version = "$Latest"
  }

  instance_refresh {
    strategy = "Rolling"
    preferences {
      min_healthy_percentage = 0
    }
  }

  tag {
    key                 = "Name"
    value               = var.instance_name
    propagate_at_launch = true
  }

  dynamic "tag" {
    for_each = var.tags
    content {
      key                 = tag.key
      value               = tag.value
      propagate_at_launch = true
    }
  }
}

# ==============================================================================
# Application Load Balancer for Neo4j
# ==============================================================================

resource "aws_lb" "neo4j" {
  name               = "${var.instance_name}-alb"
  internal           = true
  load_balancer_type = "application"
  security_groups    = [var.security_group_id]
  subnets            = var.subnet_ids

  enable_deletion_protection = var.enable_deletion_protection

  tags = var.tags
}

# Target Group for Bolt Protocol (7687)
resource "aws_lb_target_group" "neo4j_bolt" {
  name     = "${var.instance_name}-bolt-tg"
  port     = 7687
  protocol = "HTTP"
  vpc_id   = var.vpc_id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/db/neo4j/tx/commit"
    port                = "7474"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = var.tags
}

# Target Group for HTTP (7474)
resource "aws_lb_target_group" "neo4j_http" {
  name     = "${var.instance_name}-http-tg"
  port     = 7474
  protocol = "HTTP"
  vpc_id   = var.vpc_id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/db/neo4j/tx/commit"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = var.tags
}

# Listener for Bolt Protocol
resource "aws_lb_listener" "neo4j_bolt" {
  load_balancer_arn = aws_lb.neo4j.arn
  port              = "7687"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.neo4j_bolt.arn
  }
}

# Listener for HTTP
resource "aws_lb_listener" "neo4j_http" {
  load_balancer_arn = aws_lb.neo4j.arn
  port              = "7474"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.neo4j_http.arn
  }
}

# ==============================================================================
# IAM Role and Instance Profile
# ==============================================================================

resource "aws_iam_role" "neo4j" {
  name = "${var.instance_name}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "neo4j" {
  name = "${var.instance_name}-policy"
  role = aws_iam_role.neo4j.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances",
          "ec2:DescribeVolumes",
          "ec2:AttachVolume",
          "ec2:DetachVolume",
          "cloudwatch:PutMetricData",
          "logs:PutLogEvents",
          "logs:CreateLogGroup",
          "logs:CreateLogStream"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "neo4j" {
  name = "${var.instance_name}-profile"
  role = aws_iam_role.neo4j.name

  tags = var.tags
}
  spec {
    replicas = 1
    selector {
      match_labels = {
        app = "neo4j"
      }
    }
    template {
      metadata {
        labels = {
          app = "neo4j"
        }
      }
      spec {
        container {
          name  = "neo4j"
          image = "neo4j:5.15"
          port {
            container_port = 7687
          }
          env {
            name  = "NEO4J_AUTH"
            value = "neo4j/password"
          }
          volume_mount {
            name      = "neo4j-data"
            mount_path = "/data"
          }
        }
        volume {
          name = "neo4j-data"
          persistent_volume_claim {
            claim_name = "neo4j-pvc"
          }
        }
      }
    }
  }
}
