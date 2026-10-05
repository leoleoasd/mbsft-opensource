#!/usr/bin/env bash
# Example launch commands. Adjust paths, parallelism, and hyperparameters to your setup.

# Single node, torchrun, 8 GPUs
uv run torchrun --nproc_per_node=8 scripts/sft_train.py \
    --model /path/to/models/Qwen3-4B-Thinking-2507 \
    --data-path /path/to/dataset \
    --output-dir /path/to/outputs/qwen3-4b-sft \
    --chat-template chat_templates/qwen3_2507_thinking.jinja \
    --tensor-model-parallel-size 4 \
    --context-parallel-size 2 \
    --sequence-parallel \
    --seq-length 131072 \
    --global-batch-size 32 \
    --micro-batch-size 1 \
    --lr 5e-5 \
    --lr-warmup-iters 0 \
    --epochs 3 \
    --save-interval 1 \
    --wandb-project my-sft-run


# Multi-node on Kubernetes. launch_ray.sh / torchrun rely on these env vars:
#   K8S_WORLD_SIZE=4
#   K8S_MASTER_PORT=23456
#   K8S_RANK=0
#   K8S_MASTER_ADDR=<hostname-of-rank-0-pod>
uv run torchrun --nnodes $K8S_WORLD_SIZE --node-rank $K8S_RANK --master-addr $K8S_MASTER_ADDR --nproc_per_node=8 scripts/sft_train.py \
    --model /path/to/models/Qwen3-4B-Thinking-2507 \
    --data-path /path/to/dataset \
    --output-dir /path/to/outputs/qwen3-4b-sft \
    --chat-template chat_templates/qwen3_2507_thinking.jinja \
    --tensor-model-parallel-size 4 \
    --context-parallel-size 2 \
    --sequence-parallel \
    --seq-length 131072 \
    --global-batch-size 32 \
    --micro-batch-size 1 \
    --lr 5e-5 \
    --lr-warmup-iters 0 \
    --epochs 3 \
    --save-interval 1 \
    --wandb-project my-sft-run


# 30B MoE with expert parallelism, multi-node
uv run torchrun --nnodes $K8S_WORLD_SIZE --node-rank $K8S_RANK --master-addr $K8S_MASTER_ADDR --nproc_per_node=8 scripts/sft_train.py \
    --model /path/to/models/Qwen3-30B-A3B-Thinking-2507 \
    --data-path /path/to/dataset \
    --output-dir /path/to/outputs/qwen3-30b-a3b-sft \
    --chat-template chat_templates/qwen3_2507_thinking.jinja \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --context-parallel-size 8 \
    --expert-model-parallel-size 8 \
    --sequence-parallel \
    --seq-length 131072 \
    --global-batch-size 32 \
    --micro-batch-size 1 \
    --lr 5e-5 \
    --lr-warmup-iters 0 \
    --epochs 3 \
    --save-interval 1 \
    --wandb-project my-sft-run


# Multi-node on Kubernetes with the KubeRay operator: see k8s/rayjob.yaml
# (RayJob submits scripts/sft_ray.py — no torchrun or launch_ray.sh needed).
