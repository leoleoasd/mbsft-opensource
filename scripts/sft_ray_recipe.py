#!/usr/bin/env python3
"""
Ray launcher that runs the FORK's own entrypoint (scripts/training/run_recipe.py)
under a Ray cluster, instead of our custom sft_train.py.

Purpose: test whether the Qwen3.5-MoE gradient explosion reproduces on the fully
official code path (official recipe config + official forward_step + official data
provider). Reuses sft_ray.py's topology/placement-group discovery; each actor sets
the standard torch.distributed env vars and calls run_recipe.main() with the recipe
args passed through.

Usage:
    ray job submit --address=auto --working-dir /path/to/mbsft -- \
        python scripts/sft_ray_recipe.py --num-nodes 8 \
            --recipe qwen35_vl_35b_a3b_pretrain_mock_config \
            --step_func qwen3_vl_step \
            --hf_path /path/to/models/Qwen3.6-35B-A3B
    (everything after the launcher's own flags is forwarded to run_recipe.py)
"""

import argparse
import logging
import os
import sys
from collections import Counter

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# Reuse the proven topology / placement-group logic from sft_ray.py
from scripts.sft_ray import (
    create_placement_group,
    _get_local_gpu_id,
)

# Path to a Megatron-Bridge fork checkout containing scripts/training/run_recipe.py
# (must be on shared storage reachable from every cluster node).
# See pyproject.toml for the pinned fork; set via env var MBSFT_FORK_REPO.
FORK_REPO = os.environ.get("MBSFT_FORK_REPO", "")


@ray.remote(num_gpus=1)
class RecipeWorker:
    """One actor per GPU. Sets torch.distributed env vars and runs run_recipe.main()."""

    def get_master_addr_and_port(self):
        import socket
        ip = ray.util.get_node_ip_address()
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        return ip, port

    def run(self, recipe_argv: list[str], env_vars: dict[str, str]) -> str:
        os.environ.update(env_vars)
        os.environ["LOCAL_RANK"] = str(_get_local_gpu_id())

        rank = env_vars["RANK"]
        local_rank = os.environ["LOCAL_RANK"]

        import torch
        torch.cuda.set_device(f"cuda:{local_rank}")

        print(
            f"[Rank {rank}] LOCAL_RANK={local_rank} "
            f"WORLD_SIZE={env_vars['WORLD_SIZE']} "
            f"MASTER_ADDR={env_vars['MASTER_ADDR']}:{env_vars['MASTER_PORT']} "
            "(run_recipe.py)",
            flush=True,
        )

        # Make the fork's scripts/training importable as run_recipe
        recipe_dir = os.path.join(FORK_REPO, "scripts", "training")
        if recipe_dir not in sys.path:
            sys.path.insert(0, recipe_dir)

        sys.argv = ["run_recipe.py", *recipe_argv]
        import run_recipe
        run_recipe.main()
        return f"[Rank {rank}] success"


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Launch the fork's run_recipe.py via Ray (one actor per GPU)",
    )
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    args, recipe_args = parser.parse_known_args()

    if not os.path.isdir(os.path.join(FORK_REPO, "scripts", "training")):
        raise SystemExit(
            f"MBSFT_FORK_REPO={FORK_REPO!r} does not contain scripts/training/run_recipe.py; "
            "point it at a Megatron-Bridge fork checkout (shared across all cluster nodes)."
        )

    ray.init(address="auto", ignore_reinit_error=True)

    available_nodes = [
        n for n in ray.nodes()
        if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0
    ]
    if not available_nodes:
        raise SystemExit("No alive GPU nodes found in Ray cluster.")

    num_nodes = args.num_nodes if args.num_nodes is not None else len(available_nodes)
    if num_nodes > len(available_nodes):
        raise SystemExit(
            f"Requested {num_nodes} nodes but only {len(available_nodes)} GPU nodes available."
        )

    world_size = num_nodes * args.gpus_per_node
    print(f"[Ray Recipe] {num_nodes} node(s), {args.gpus_per_node} GPUs/node, world_size={world_size}")
    print(f"[Ray Recipe] run_recipe.py args: {recipe_args}")

    pg, reordered_indices, node_ips = create_placement_group(world_size)
    node_rank_counts = Counter(node_ips)

    actors = []
    master_addr, master_port = None, None
    for rank in range(world_size):
        actor = RecipeWorker.options(
            num_gpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=reordered_indices[rank],
            ),
        ).remote()
        if rank == 0:
            master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            print(f"[Ray Recipe] master={master_addr}:{master_port}")
        actors.append(actor)

    seen_nodes: dict[str, int] = {}
    futures = []
    for rank, actor in enumerate(actors):
        node_ip = node_ips[rank]
        seen_nodes[node_ip] = seen_nodes.get(node_ip, 0) + 1
        env_vars = {
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(node_rank_counts[node_ip]),
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }
        futures.append(actor.run.remote(recipe_args, env_vars))

    results = ray.get(futures)
    for r in results:
        print(r, flush=True)
    print("[Ray Recipe] All ranks finished successfully.", flush=True)


if __name__ == "__main__":
    main()
