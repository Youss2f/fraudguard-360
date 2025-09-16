provider "kubernetes" {
  config_path = "~/.kube/config"
}

module "k8s_namespace" {
  source = "./modules/k8s"
}

module "kafka" {
  source = "./modules/kafka"
}

module "neo4j" {
  source = "./modules/neo4j"
}

# module "monitoring" {
#   source = "prometheus-community/prometheus/kubernetes"
#   version = "0.11.0"
#   # Config options
# }
