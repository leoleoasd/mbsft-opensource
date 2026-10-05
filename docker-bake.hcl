group "default" {
  targets = ["megatron_bridge"]
}

# Registry/image to push to. Override without editing this file:
#   REPO=ghcr.io/you/mbsft docker buildx bake
variable "REPO" {
  default = "ghcr.io/CHANGE-ME/mbsft"
}

# Training base image (must provide CUDA toolkit + drivers headers).
variable "BASE_IMAGE" {
  default = "nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04"
}

target "megatron_bridge" {
  context = "."
  dockerfile = "Dockerfile"
  args = {
    BASE_IMAGE = BASE_IMAGE
  }
  tags = ["${REPO}:megatron_bridge"]
  platforms = ["linux/amd64"]
  push = true
}
