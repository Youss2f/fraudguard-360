resource "kubernetes_persistent_volume_claim" "neo4j_pvc" {
  metadata {
    name      = "neo4j-pvc"
    namespace = "fraudguard"
  }
  spec {
    access_modes = ["ReadWriteOnce"]
    resources {
      requests = {
        storage = "50Gi"
      }
    }
  }
}

resource "kubernetes_deployment" "neo4j" {
  metadata {
    name      = "neo4j"
    namespace = "fraudguard"
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
