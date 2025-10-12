# ==============================================================================
# FraudGuard 360 - Monitoring Module Variables
# Input variables for the monitoring module
# ==============================================================================

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "common_tags" {
  description = "Common tags to be applied to all resources"
  type        = map(string)
  default     = {}
}

# Prometheus Configuration
variable "prometheus_image" {
  description = "Prometheus Docker image"
  type        = string
  default     = "prom/prometheus:v2.45.0"
}

variable "prometheus_replicas" {
  description = "Number of Prometheus replicas"
  type        = number
  default     = 1
  
  validation {
    condition     = var.prometheus_replicas >= 1
    error_message = "Prometheus replicas must be at least 1."
  }
}

variable "prometheus_retention" {
  description = "Prometheus data retention period"
  type        = string
  default     = "15d"
}

variable "prometheus_cpu_request" {
  description = "Prometheus CPU request"
  type        = string
  default     = "500m"
}

variable "prometheus_cpu_limit" {
  description = "Prometheus CPU limit"
  type        = string
  default     = "2000m"
}

variable "prometheus_memory_request" {
  description = "Prometheus memory request"
  type        = string
  default     = "1Gi"
}

variable "prometheus_memory_limit" {
  description = "Prometheus memory limit"
  type        = string
  default     = "4Gi"
}

# Grafana Configuration
variable "grafana_image" {
  description = "Grafana Docker image"
  type        = string
  default     = "grafana/grafana:10.0.0"
}

variable "grafana_replicas" {
  description = "Number of Grafana replicas"
  type        = number
  default     = 1
  
  validation {
    condition     = var.grafana_replicas >= 1
    error_message = "Grafana replicas must be at least 1."
  }
}

variable "grafana_admin_user" {
  description = "Grafana admin username"
  type        = string
  default     = "admin"
}

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  default     = "admin"
  sensitive   = true
}

variable "grafana_service_type" {
  description = "Grafana service type"
  type        = string
  default     = "ClusterIP"
  
  validation {
    condition     = contains(["ClusterIP", "NodePort", "LoadBalancer"], var.grafana_service_type)
    error_message = "Service type must be one of: ClusterIP, NodePort, LoadBalancer."
  }
}

variable "grafana_cpu_request" {
  description = "Grafana CPU request"
  type        = string
  default     = "250m"
}

variable "grafana_cpu_limit" {
  description = "Grafana CPU limit"
  type        = string
  default     = "500m"
}

variable "grafana_memory_request" {
  description = "Grafana memory request"
  type        = string
  default     = "256Mi"
}

variable "grafana_memory_limit" {
  description = "Grafana memory limit"
  type        = string
  default     = "512Mi"
}

# AlertManager Configuration
variable "alertmanager_image" {
  description = "AlertManager Docker image"
  type        = string
  default     = "prom/alertmanager:v0.25.0"
}

variable "alertmanager_replicas" {
  description = "Number of AlertManager replicas"
  type        = number
  default     = 1
  
  validation {
    condition     = var.alertmanager_replicas >= 1
    error_message = "AlertManager replicas must be at least 1."
  }
}

variable "alertmanager_cpu_request" {
  description = "AlertManager CPU request"
  type        = string
  default     = "100m"
}

variable "alertmanager_cpu_limit" {
  description = "AlertManager CPU limit"
  type        = string
  default     = "200m"
}

variable "alertmanager_memory_request" {
  description = "AlertManager memory request"
  type        = string
  default     = "128Mi"
}

variable "alertmanager_memory_limit" {
  description = "AlertManager memory limit"
  type        = string
  default     = "256Mi"
}

# Alert Configuration
variable "smtp_smarthost" {
  description = "SMTP server for email alerts"
  type        = string
  default     = "localhost:587"
}

variable "smtp_from" {
  description = "From email address for alerts"
  type        = string
  default     = "alerts@fraudguard360.com"
}

variable "alert_email_configs" {
  description = "Email configurations for alerts"
  type = list(object({
    to      = string
    subject = string
    body    = string
  }))
  default = [
    {
      to      = "admin@fraudguard360.com"
      subject = "FraudGuard 360 Alert: {{ .GroupLabels.alertname }}"
      body    = "Alert: {{ .GroupLabels.alertname }}\nSeverity: {{ .CommonLabels.severity }}\nDescription: {{ .CommonAnnotations.description }}"
    }
  ]
}

variable "alert_webhook_configs" {
  description = "Webhook configurations for alerts"
  type = list(object({
    url = string
  }))
  default = []
}

# Storage Configuration
variable "enable_persistent_storage" {
  description = "Enable persistent storage for monitoring components"
  type        = bool
  default     = false
}

variable "prometheus_storage_class" {
  description = "Storage class for Prometheus persistent volume"
  type        = string
  default     = "gp2"
}

variable "prometheus_storage_size" {
  description = "Storage size for Prometheus persistent volume"
  type        = string
  default     = "50Gi"
}

variable "grafana_storage_class" {
  description = "Storage class for Grafana persistent volume"
  type        = string
  default     = "gp2"
}

variable "grafana_storage_size" {
  description = "Storage size for Grafana persistent volume"
  type        = string
  default     = "10Gi"
}

# Security Configuration
variable "enable_security_context" {
  description = "Enable security context for containers"
  type        = bool
  default     = true
}

variable "run_as_non_root" {
  description = "Run containers as non-root user"
  type        = bool
  default     = true
}

variable "run_as_user" {
  description = "User ID to run containers as"
  type        = number
  default     = 65534
}

variable "fs_group" {
  description = "File system group for volumes"
  type        = number
  default     = 65534
}

# Network Policy
variable "enable_network_policies" {
  description = "Enable network policies for monitoring namespace"
  type        = bool
  default     = false
}

# Custom Dashboards
variable "custom_dashboards" {
  description = "List of custom Grafana dashboards to deploy"
  type = list(object({
    name = string
    json = string
  }))
  default = []
}

# Service Monitor Configuration
variable "enable_service_monitors" {
  description = "Enable ServiceMonitor resources for service discovery"
  type        = bool
  default     = false
}

# External Services
variable "external_prometheus_url" {
  description = "External Prometheus URL for federation"
  type        = string
  default     = ""
}

variable "external_alertmanager_url" {
  description = "External AlertManager URL"
  type        = string
  default     = ""
}