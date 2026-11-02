
terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

provider "kubernetes" {
  config_path    = pathexpand("~/.kube/config")
  config_context = "minikube"
}

resource "kubernetes_namespace" "holiday" {
  metadata {
    name = "holiday-mlops"
  }
}

output "namespace_name" {
  value = kubernetes_namespace.holiday.metadata[0].name
}
