resource "kubernetes_namespace" "fraudguard" {
  metadata {
    name = "fraudguard"
  }
}
