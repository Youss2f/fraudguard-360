# Use Helm provider
provider "helm" {
  kubernetes = {
    config_path = "~/.kube/config"
  }
}

resource "helm_release" "kafka" {
  name       = "kafka"
  repository = "https://charts.bitnami.com/bitnami"
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
