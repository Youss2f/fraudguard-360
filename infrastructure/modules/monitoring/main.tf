# ==============================================================================
# FraudGuard 360 - Monitoring Module
# Comprehensive monitoring and observability stack
# ==============================================================================

# Prometheus Namespace
resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = "monitoring"
    labels = {
      name = "monitoring"
      app  = "prometheus"
    }
  }
}

# Prometheus ServiceAccount
resource "kubernetes_service_account" "prometheus" {
  metadata {
    name      = "prometheus"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
  }
}

# Prometheus ClusterRole
resource "kubernetes_cluster_role" "prometheus" {
  metadata {
    name = "prometheus"
  }

  rule {
    api_groups = [""]
    resources  = ["nodes", "nodes/proxy", "services", "endpoints", "pods"]
    verbs      = ["get", "list", "watch"]
  }

  rule {
    api_groups = ["extensions"]
    resources  = ["ingresses"]
    verbs      = ["get", "list", "watch"]
  }

  rule {
    non_resource_urls = ["/metrics"]
    verbs             = ["get"]
  }
}

# Prometheus ClusterRoleBinding
resource "kubernetes_cluster_role_binding" "prometheus" {
  metadata {
    name = "prometheus"
  }

  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "ClusterRole"
    name      = kubernetes_cluster_role.prometheus.metadata[0].name
  }

  subject {
    kind      = "ServiceAccount"
    name      = kubernetes_service_account.prometheus.metadata[0].name
    namespace = kubernetes_namespace.monitoring.metadata[0].name
  }
}

# Prometheus ConfigMap
resource "kubernetes_config_map" "prometheus_config" {
  metadata {
    name      = "prometheus-config"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
  }

  data = {
    "prometheus.yml" = yamlencode({
      global = {
        scrape_interval     = "15s"
        evaluation_interval = "15s"
      }

      rule_files = [
        "/etc/prometheus/rules/*.yml"
      ]

      alerting = {
        alertmanagers = [
          {
            static_configs = [
              {
                targets = ["alertmanager:9093"]
              }
            ]
          }
        ]
      }

      scrape_configs = [
        {
          job_name = "prometheus"
          static_configs = [
            {
              targets = ["localhost:9090"]
            }
          ]
        },
        {
          job_name = "kubernetes-apiservers"
          kubernetes_sd_configs = [
            {
              role = "endpoints"
            }
          ]
          scheme = "https"
          tls_config = {
            ca_file = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
          }
          bearer_token_file = "/var/run/secrets/kubernetes.io/serviceaccount/token"
          relabel_configs = [
            {
              source_labels = ["__meta_kubernetes_namespace", "__meta_kubernetes_service_name", "__meta_kubernetes_endpoint_port_name"]
              action        = "keep"
              regex         = "default;kubernetes;https"
            }
          ]
        },
        {
          job_name = "kubernetes-nodes"
          kubernetes_sd_configs = [
            {
              role = "node"
            }
          ]
          scheme = "https"
          tls_config = {
            ca_file = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
          }
          bearer_token_file = "/var/run/secrets/kubernetes.io/serviceaccount/token"
          relabel_configs = [
            {
              action = "labelmap"
              regex  = "__meta_kubernetes_node_label_(.+)"
            },
            {
              target_label = "__address__"
              replacement  = "kubernetes.default.svc:443"
            },
            {
              source_labels = ["__meta_kubernetes_node_name"]
              regex         = "(.+)"
              target_label  = "__metrics_path__"
              replacement   = "/api/v1/nodes/$${1}/proxy/metrics"
            }
          ]
        },
        {
          job_name = "kubernetes-cadvisor"
          kubernetes_sd_configs = [
            {
              role = "node"
            }
          ]
          scheme = "https"
          tls_config = {
            ca_file = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
          }
          bearer_token_file = "/var/run/secrets/kubernetes.io/serviceaccount/token"
          relabel_configs = [
            {
              action = "labelmap"
              regex  = "__meta_kubernetes_node_label_(.+)"
            },
            {
              target_label = "__address__"
              replacement  = "kubernetes.default.svc:443"
            },
            {
              source_labels = ["__meta_kubernetes_node_name"]
              regex         = "(.+)"
              target_label  = "__metrics_path__"
              replacement   = "/api/v1/nodes/$${1}/proxy/metrics/cadvisor"
            }
          ]
        },
        {
          job_name = "kubernetes-service-endpoints"
          kubernetes_sd_configs = [
            {
              role = "endpoints"
            }
          ]
          relabel_configs = [
            {
              source_labels = ["__meta_kubernetes_service_annotation_prometheus_io_scrape"]
              action        = "keep"
              regex         = true
            },
            {
              source_labels = ["__meta_kubernetes_service_annotation_prometheus_io_scheme"]
              action        = "replace"
              target_label  = "__scheme__"
              regex         = "(https?)"
            },
            {
              source_labels = ["__meta_kubernetes_service_annotation_prometheus_io_path"]
              action        = "replace"
              target_label  = "__metrics_path__"
              regex         = "(.+)"
            },
            {
              source_labels = ["__address__", "__meta_kubernetes_service_annotation_prometheus_io_port"]
              action        = "replace"
              target_label  = "__address__"
              regex         = "([^:]+)(?::\\d+)?;(\\d+)"
              replacement   = "$1:$2"
            },
            {
              action = "labelmap"
              regex  = "__meta_kubernetes_service_label_(.+)"
            },
            {
              source_labels = ["__meta_kubernetes_namespace"]
              action        = "replace"
              target_label  = "kubernetes_namespace"
            },
            {
              source_labels = ["__meta_kubernetes_service_name"]
              action        = "replace"
              target_label  = "kubernetes_name"
            }
          ]
        },
        {
          job_name = "kubernetes-pods"
          kubernetes_sd_configs = [
            {
              role = "pod"
            }
          ]
          relabel_configs = [
            {
              source_labels = ["__meta_kubernetes_pod_annotation_prometheus_io_scrape"]
              action        = "keep"
              regex         = true
            },
            {
              source_labels = ["__meta_kubernetes_pod_annotation_prometheus_io_path"]
              action        = "replace"
              target_label  = "__metrics_path__"
              regex         = "(.+)"
            },
            {
              source_labels = ["__address__", "__meta_kubernetes_pod_annotation_prometheus_io_port"]
              action        = "replace"
              regex         = "([^:]+)(?::\\d+)?;(\\d+)"
              replacement   = "$1:$2"
              target_label  = "__address__"
            },
            {
              action = "labelmap"
              regex  = "__meta_kubernetes_pod_label_(.+)"
            },
            {
              source_labels = ["__meta_kubernetes_namespace"]
              action        = "replace"
              target_label  = "kubernetes_namespace"
            },
            {
              source_labels = ["__meta_kubernetes_pod_name"]
              action        = "replace"
              target_label  = "kubernetes_pod_name"
            }
          ]
        }
      ]
    })
  }
}

# Prometheus AlertManager Rules
resource "kubernetes_config_map" "prometheus_rules" {
  metadata {
    name      = "prometheus-rules"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
  }

  data = {
    "fraud-detection.yml" = yamlencode({
      groups = [
        {
          name = "fraud-detection"
          rules = [
            {
              alert = "HighFraudDetectionLatency"
              expr  = "histogram_quantile(0.95, sum(rate(fraud_detection_duration_seconds_bucket[5m])) by (le)) > 0.5"
              for   = "2m"
              labels = {
                severity = "warning"
              }
              annotations = {
                summary     = "High fraud detection latency"
                description = "95th percentile latency is above 500ms for {{ $labels.job }}"
              }
            },
            {
              alert = "FraudDetectionErrors"
              expr  = "rate(fraud_detection_errors_total[5m]) > 0.01"
              for   = "1m"
              labels = {
                severity = "critical"
              }
              annotations = {
                summary     = "High fraud detection error rate"
                description = "Error rate is above 1% for {{ $labels.job }}"
              }
            },
            {
              alert = "ModelAccuracyDegraded"
              expr  = "fraud_model_accuracy < 0.85"
              for   = "5m"
              labels = {
                severity = "warning"
              }
              annotations = {
                summary     = "Model accuracy degraded"
                description = "Model accuracy has dropped below 85%"
              }
            }
          ]
        },
        {
          name = "kubernetes"
          rules = [
            {
              alert = "KubernetesPodCrashLooping"
              expr  = "rate(kube_pod_container_status_restarts_total[15m]) > 0"
              for   = "5m"
              labels = {
                severity = "warning"
              }
              annotations = {
                summary     = "Pod is crash looping"
                description = "Pod {{ $labels.namespace }}/{{ $labels.pod }} is crash looping"
              }
            },
            {
              alert = "KubernetesNodeNotReady"
              expr  = "kube_node_status_condition{condition=\"Ready\",status=\"true\"} == 0"
              for   = "5m"
              labels = {
                severity = "critical"
              }
              annotations = {
                summary     = "Node not ready"
                description = "Node {{ $labels.node }} is not ready"
              }
            }
          ]
        }
      ]
    })
  }
}

# Prometheus Deployment
resource "kubernetes_deployment" "prometheus" {
  metadata {
    name      = "prometheus"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
    labels = {
      app = "prometheus"
    }
  }

  spec {
    replicas = var.prometheus_replicas

    selector {
      match_labels = {
        app = "prometheus"
      }
    }

    template {
      metadata {
        labels = {
          app = "prometheus"
        }
      }

      spec {
        service_account_name = kubernetes_service_account.prometheus.metadata[0].name

        container {
          name  = "prometheus"
          image = var.prometheus_image
          
          port {
            container_port = 9090
          }

          args = [
            "--config.file=/etc/prometheus/prometheus.yml",
            "--storage.tsdb.path=/prometheus/",
            "--web.console.libraries=/etc/prometheus/console_libraries",
            "--web.console.templates=/etc/prometheus/consoles",
            "--storage.tsdb.retention.time=${var.prometheus_retention}",
            "--web.enable-lifecycle",
            "--web.enable-admin-api"
          ]

          volume_mount {
            name       = "prometheus-config-volume"
            mount_path = "/etc/prometheus/"
          }

          volume_mount {
            name       = "prometheus-rules-volume"
            mount_path = "/etc/prometheus/rules/"
          }

          volume_mount {
            name       = "prometheus-storage-volume"
            mount_path = "/prometheus/"
          }

          resources {
            limits = {
              cpu    = var.prometheus_cpu_limit
              memory = var.prometheus_memory_limit
            }
            requests = {
              cpu    = var.prometheus_cpu_request
              memory = var.prometheus_memory_request
            }
          }

          liveness_probe {
            http_get {
              path = "/-/healthy"
              port = 9090
            }
            initial_delay_seconds = 30
            period_seconds        = 15
          }

          readiness_probe {
            http_get {
              path = "/-/ready"
              port = 9090
            }
            initial_delay_seconds = 30
            period_seconds        = 5
          }
        }

        volume {
          name = "prometheus-config-volume"
          config_map {
            default_mode = "0420"
            name         = kubernetes_config_map.prometheus_config.metadata[0].name
          }
        }

        volume {
          name = "prometheus-rules-volume"
          config_map {
            default_mode = "0420"
            name         = kubernetes_config_map.prometheus_rules.metadata[0].name
          }
        }

        volume {
          name = "prometheus-storage-volume"
          empty_dir {}
        }
      }
    }
  }
}

# Prometheus Service
resource "kubernetes_service" "prometheus" {
  metadata {
    name      = "prometheus"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
    labels = {
      app = "prometheus"
    }
  }

  spec {
    selector = {
      app = "prometheus"
    }

    port {
      port        = 9090
      target_port = 9090
      protocol    = "TCP"
    }

    type = "ClusterIP"
  }
}

# Grafana ConfigMap
resource "kubernetes_config_map" "grafana_config" {
  metadata {
    name      = "grafana-config"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
  }

  data = {
    "grafana.ini" = <<EOF
[analytics]
check_for_updates = true

[grafana_net]
url = https://grafana.net

[log]
mode = console

[paths]
data = /var/lib/grafana/data
logs = /var/log/grafana
plugins = /var/lib/grafana/plugins
provisioning = /etc/grafana/provisioning

[security]
admin_user = ${var.grafana_admin_user}
admin_password = ${var.grafana_admin_password}

[server]
http_port = 3000
root_url = http://localhost:3000/
EOF
  }
}

# Grafana Datasources
resource "kubernetes_config_map" "grafana_datasources" {
  metadata {
    name      = "grafana-datasources"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
  }

  data = {
    "datasources.yaml" = yamlencode({
      apiVersion = 1
      datasources = [
        {
          name      = "Prometheus"
          type      = "prometheus"
          url       = "http://prometheus:9090"
          access    = "proxy"
          isDefault = true
        }
      ]
    })
  }
}

# Grafana Dashboards ConfigMap
resource "kubernetes_config_map" "grafana_dashboards_config" {
  metadata {
    name      = "grafana-dashboards-config" 
    namespace = kubernetes_namespace.monitoring.metadata[0].name
  }

  data = {
    "dashboards.yaml" = yamlencode({
      apiVersion = 1
      providers = [
        {
          name            = "default"
          orgId           = 1
          folder          = ""
          type            = "file"
          disableDeletion = false
          updateIntervalSeconds = 10
          allowUiUpdates  = true
          options = {
            path = "/var/lib/grafana/dashboards"
          }
        }
      ]
    })
  }
}

# Grafana Deployment
resource "kubernetes_deployment" "grafana" {
  metadata {
    name      = "grafana"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
    labels = {
      app = "grafana"
    }
  }

  spec {
    replicas = var.grafana_replicas

    selector {
      match_labels = {
        app = "grafana"
      }
    }

    template {
      metadata {
        labels = {
          app = "grafana"
        }
      }

      spec {
        container {
          name  = "grafana"
          image = var.grafana_image

          port {
            container_port = 3000
          }

          env {
            name  = "GF_SECURITY_ADMIN_USER"
            value = var.grafana_admin_user
          }

          env {
            name  = "GF_SECURITY_ADMIN_PASSWORD"
            value = var.grafana_admin_password
          }

          volume_mount {
            name       = "grafana-config"
            mount_path = "/etc/grafana/grafana.ini"
            sub_path   = "grafana.ini"
          }

          volume_mount {
            name       = "grafana-datasources"
            mount_path = "/etc/grafana/provisioning/datasources"
          }

          volume_mount {
            name       = "grafana-dashboards-config"
            mount_path = "/etc/grafana/provisioning/dashboards"
          }

          volume_mount {
            name       = "grafana-storage"
            mount_path = "/var/lib/grafana"
          }

          resources {
            limits = {
              cpu    = var.grafana_cpu_limit
              memory = var.grafana_memory_limit
            }
            requests = {
              cpu    = var.grafana_cpu_request
              memory = var.grafana_memory_request
            }
          }

          liveness_probe {
            http_get {
              path = "/api/health"
              port = 3000
            }
            initial_delay_seconds = 60
            period_seconds        = 10
          }

          readiness_probe {
            http_get {
              path = "/api/health"
              port = 3000
            }
            initial_delay_seconds = 30
            period_seconds        = 5
          }
        }

        volume {
          name = "grafana-config"
          config_map {
            name = kubernetes_config_map.grafana_config.metadata[0].name
          }
        }

        volume {
          name = "grafana-datasources"
          config_map {
            name = kubernetes_config_map.grafana_datasources.metadata[0].name
          }
        }

        volume {
          name = "grafana-dashboards-config"
          config_map {
            name = kubernetes_config_map.grafana_dashboards_config.metadata[0].name
          }
        }

        volume {
          name = "grafana-storage"
          empty_dir {}
        }
      }
    }
  }
}

# Grafana Service
resource "kubernetes_service" "grafana" {
  metadata {
    name      = "grafana"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
    labels = {
      app = "grafana"
    }
  }

  spec {
    selector = {
      app = "grafana"
    }

    port {
      port        = 3000
      target_port = 3000
      protocol    = "TCP"
    }

    type = var.grafana_service_type
  }
}

# AlertManager ConfigMap
resource "kubernetes_config_map" "alertmanager_config" {
  metadata {
    name      = "alertmanager-config"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
  }

  data = {
    "alertmanager.yml" = yamlencode({
      global = {
        smtp_smarthost = var.smtp_smarthost
        smtp_from      = var.smtp_from
      }

      route = {
        group_by        = ["alertname"]
        group_wait      = "10s"
        group_interval  = "10s"
        repeat_interval = "1h"
        receiver        = "web.hook"
      }

      receivers = [
        {
          name = "web.hook"
          email_configs = var.alert_email_configs
          webhook_configs = var.alert_webhook_configs
        }
      ]

      inhibit_rules = [
        {
          source_match = {
            severity = "critical"
          }
          target_match = {
            severity = "warning"
          }
          equal = ["alertname", "dev", "instance"]
        }
      ]
    })
  }
}

# AlertManager Deployment
resource "kubernetes_deployment" "alertmanager" {
  metadata {
    name      = "alertmanager"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
    labels = {
      app = "alertmanager"
    }
  }

  spec {
    replicas = var.alertmanager_replicas

    selector {
      match_labels = {
        app = "alertmanager"
      }
    }

    template {
      metadata {
        labels = {
          app = "alertmanager"
        }
      }

      spec {
        container {
          name  = "alertmanager"
          image = var.alertmanager_image

          port {
            container_port = 9093
          }

          args = [
            "--config.file=/etc/alertmanager/alertmanager.yml",
            "--storage.path=/alertmanager",
            "--web.external-url=http://localhost:9093/"
          ]

          volume_mount {
            name       = "alertmanager-config"
            mount_path = "/etc/alertmanager"
          }

          volume_mount {
            name       = "alertmanager-storage"
            mount_path = "/alertmanager"
          }

          resources {
            limits = {
              cpu    = var.alertmanager_cpu_limit
              memory = var.alertmanager_memory_limit
            }
            requests = {
              cpu    = var.alertmanager_cpu_request
              memory = var.alertmanager_memory_request
            }
          }

          liveness_probe {
            http_get {
              path = "/-/healthy"
              port = 9093
            }
            initial_delay_seconds = 30
            period_seconds        = 15
          }

          readiness_probe {
            http_get {
              path = "/-/ready"
              port = 9093
            }
            initial_delay_seconds = 30
            period_seconds        = 5
          }
        }

        volume {
          name = "alertmanager-config"
          config_map {
            name = kubernetes_config_map.alertmanager_config.metadata[0].name
          }
        }

        volume {
          name = "alertmanager-storage"
          empty_dir {}
        }
      }
    }
  }
}

# AlertManager Service
resource "kubernetes_service" "alertmanager" {
  metadata {
    name      = "alertmanager"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
    labels = {
      app = "alertmanager"
    }
  }

  spec {
    selector = {
      app = "alertmanager"
    }

    port {
      port        = 9093
      target_port = 9093
      protocol    = "TCP"
    }

    type = "ClusterIP"
  }
}