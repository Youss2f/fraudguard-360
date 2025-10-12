# ==============================================================================
# FraudGuard 360 - Monitoring Module Outputs
# Output values from monitoring stack deployment
# ==============================================================================

output "monitoring_namespace" {
  description = "Monitoring namespace name"
  value       = kubernetes_namespace.monitoring.metadata[0].name
}

output "prometheus_service_name" {
  description = "Prometheus service name"
  value       = kubernetes_service.prometheus.metadata[0].name
}

output "prometheus_service_port" {
  description = "Prometheus service port"
  value       = kubernetes_service.prometheus.spec[0].port[0].port
}

output "prometheus_endpoint" {
  description = "Prometheus endpoint URL"
  value       = "http://${kubernetes_service.prometheus.metadata[0].name}.${kubernetes_namespace.monitoring.metadata[0].name}.svc.cluster.local:${kubernetes_service.prometheus.spec[0].port[0].port}"
}

output "grafana_service_name" {
  description = "Grafana service name"
  value       = kubernetes_service.grafana.metadata[0].name
}

output "grafana_service_port" {
  description = "Grafana service port"
  value       = kubernetes_service.grafana.spec[0].port[0].port
}

output "grafana_endpoint" {
  description = "Grafana endpoint URL"
  value       = "http://${kubernetes_service.grafana.metadata[0].name}.${kubernetes_namespace.monitoring.metadata[0].name}.svc.cluster.local:${kubernetes_service.grafana.spec[0].port[0].port}"
}

output "grafana_admin_credentials" {
  description = "Grafana admin credentials"
  value = {
    username = var.grafana_admin_user
    password = var.grafana_admin_password
  }
  sensitive = true
}

output "alertmanager_service_name" {
  description = "AlertManager service name"
  value       = kubernetes_service.alertmanager.metadata[0].name
}

output "alertmanager_service_port" {
  description = "AlertManager service port"
  value       = kubernetes_service.alertmanager.spec[0].port[0].port
}

output "alertmanager_endpoint" {
  description = "AlertManager endpoint URL"
  value       = "http://${kubernetes_service.alertmanager.metadata[0].name}.${kubernetes_namespace.monitoring.metadata[0].name}.svc.cluster.local:${kubernetes_service.alertmanager.spec[0].port[0].port}"
}

output "monitoring_services" {
  description = "All monitoring service endpoints"
  value = {
    prometheus = {
      name     = kubernetes_service.prometheus.metadata[0].name
      port     = kubernetes_service.prometheus.spec[0].port[0].port
      endpoint = "http://${kubernetes_service.prometheus.metadata[0].name}.${kubernetes_namespace.monitoring.metadata[0].name}.svc.cluster.local:${kubernetes_service.prometheus.spec[0].port[0].port}"
    }
    grafana = {
      name     = kubernetes_service.grafana.metadata[0].name
      port     = kubernetes_service.grafana.spec[0].port[0].port
      endpoint = "http://${kubernetes_service.grafana.metadata[0].name}.${kubernetes_namespace.monitoring.metadata[0].name}.svc.cluster.local:${kubernetes_service.grafana.spec[0].port[0].port}"
    }
    alertmanager = {
      name     = kubernetes_service.alertmanager.metadata[0].name
      port     = kubernetes_service.alertmanager.spec[0].port[0].port
      endpoint = "http://${kubernetes_service.alertmanager.metadata[0].name}.${kubernetes_namespace.monitoring.metadata[0].name}.svc.cluster.local:${kubernetes_service.alertmanager.spec[0].port[0].port}"
    }
  }
}

output "prometheus_config_map" {
  description = "Prometheus configuration ConfigMap name"
  value       = kubernetes_config_map.prometheus_config.metadata[0].name
}

output "prometheus_rules_config_map" {
  description = "Prometheus rules ConfigMap name"
  value       = kubernetes_config_map.prometheus_rules.metadata[0].name
}

output "grafana_config_map" {
  description = "Grafana configuration ConfigMap name"
  value       = kubernetes_config_map.grafana_config.metadata[0].name
}

output "grafana_datasources_config_map" {
  description = "Grafana datasources ConfigMap name"
  value       = kubernetes_config_map.grafana_datasources.metadata[0].name
}

output "alertmanager_config_map" {
  description = "AlertManager configuration ConfigMap name"
  value       = kubernetes_config_map.alertmanager_config.metadata[0].name
}

output "deployment_info" {
  description = "Deployment information for monitoring components"
  value = {
    prometheus = {
      name     = kubernetes_deployment.prometheus.metadata[0].name
      replicas = kubernetes_deployment.prometheus.spec[0].replicas
      image    = var.prometheus_image
    }
    grafana = {
      name     = kubernetes_deployment.grafana.metadata[0].name
      replicas = kubernetes_deployment.grafana.spec[0].replicas
      image    = var.grafana_image
    }
    alertmanager = {
      name     = kubernetes_deployment.alertmanager.metadata[0].name
      replicas = kubernetes_deployment.alertmanager.spec[0].replicas
      image    = var.alertmanager_image
    }
  }
}

output "service_account_info" {
  description = "Service account information"
  value = {
    prometheus = {
      name      = kubernetes_service_account.prometheus.metadata[0].name
      namespace = kubernetes_service_account.prometheus.metadata[0].namespace
    }
  }
}

output "rbac_info" {
  description = "RBAC configuration information"
  value = {
    cluster_role = {
      name = kubernetes_cluster_role.prometheus.metadata[0].name
    }
    cluster_role_binding = {
      name = kubernetes_cluster_role_binding.prometheus.metadata[0].name
    }
  }
}