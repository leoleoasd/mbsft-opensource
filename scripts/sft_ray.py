#!/usr/bin/env python3
"""
Ray Job wrapper for multi-node SFT training.

Spawns one Ray actor per GPU across the cluster using a single PlacementGroup
with PACK strategy. Uses InfoActors to discover node/GPU topology, then
reorders bundle indices so consecutive ranks land on the same node
(matching torchrun's LOCAL_RANK behavior).

Each actor sets torch.distributed env vars and calls sft_train.main() directly.
No torchrun needed.

Usage:
    ray job submit --address=auto --working-dir /path/to/mbsft -- \\
        python scripts/sft_ray.py \\
            --model /path/to/models/Llama-3.1-8B-Instruct \\
            --data-path /path/to/dataset \\
            --output-dir /path/to/outputs/llama-3.1-8b-sft \\
            --chat-template chat_templates/llama3.1.jinja \\
            --tensor-model-parallel-size 2 \\
            --context-parallel-size 8 \\
            --sequence-parallel \\
            --seq-length 131072 \\
            --global-batch-size 32 \\
            --micro-batch-size 1 \\
            --lr 5e-5 \\
            --lr-warmup-iters 0 \\
            --epochs 3 \\
            --save-interval 1 \\
            --wandb-project my-sft-run

    Ray-specific options (parsed before forwarding the rest to sft_train.py):
        --gpus-per-node   Number of GPUs per node (default: 8)
        --master-port     Port for torch.distributed rendezvous (default: 29500)
"""

import argparse
import logging
import os
import socket
import subprocess
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Topology discovery (adapted from slime)
# ---------------------------------------------------------------------------

@ray.remote(num_gpus=1)
class InfoActor:
    """Temporary actor to discover which node/GPU a bundle landed on."""

    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def _sort_key(x, gpu_counts):
    """Sort bundles: nodes with more GPUs first, then by IP, then by GPU id."""
    _index, node_ip, gpu_id = x
    try:
        ip_parts = list(map(int, node_ip.split(".")))
    except ValueError:
        try:
            ip_parts = list(map(int, socket.gethostbyname(node_ip).split(".")))
        except (socket.gaierror, TypeError):
            ip_parts = [ord(c) for c in node_ip]
    return (-gpu_counts[node_ip], ip_parts, gpu_id)


def create_placement_group(num_gpus: int):
    """Create a PACK placement group and return reordered bundle indices.

    Returns:
        (pg, reordered_bundle_indices, node_ips_per_rank)
        where reordered_bundle_indices[logical_rank] = physical_bundle_index
    """
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())

    # Probe each bundle to discover (node_ip, gpu_id)
    info_actors = [
        InfoActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i,
            )
        ).remote()
        for i in range(num_gpus)
    ]
    gpu_infos = ray.get([a.get_ip_and_gpu_id.remote() for a in info_actors])
    for a in info_actors:
        ray.kill(a)

    # Sort so consecutive ranks are on the same node
    bundle_infos = [(i, gpu_infos[i][0], gpu_infos[i][1]) for i in range(num_gpus)]
    gpu_counts = Counter(ip for ip, _ in gpu_infos)
    sorted_infos = sorted(bundle_infos, key=partial(_sort_key, gpu_counts=gpu_counts))

    reordered_bundle_indices = [info[0] for info in sorted_infos]
    node_ips = [gpu_infos[info[0]][0] for info in sorted_infos]

    for logical_rank, info in enumerate(sorted_infos):
        actual_idx = info[0]
        logger.info(
            f"  rank {logical_rank:3d} -> bundle {actual_idx:3d}, "
            f"node={gpu_infos[actual_idx][0]}, gpu={gpu_infos[actual_idx][1]}"
        )

    return pg, reordered_bundle_indices, node_ips


# ---------------------------------------------------------------------------
# Worker actor
# ---------------------------------------------------------------------------

def _get_local_gpu_id():
    """Derive LOCAL_RANK from Ray's CUDA_VISIBLE_DEVICES assignment."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is None:
        return ray.get_gpu_ids()[0]
    return cvd.split(",").index(str(ray.get_gpu_ids()[0]))


@ray.remote(num_gpus=1)
class SFTWorker:
    """One actor per GPU. Sets torch.distributed env vars and runs sft_train.main()."""

    def get_master_addr_and_port(self):
        """Called on rank 0 to discover master address."""
        ip = ray.util.get_node_ip_address()
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        return ip, port

    def run(self, sft_argv: list[str], env_vars: dict[str, str]) -> str:
        os.environ.update(env_vars)
        os.environ["LOCAL_RANK"] = str(_get_local_gpu_id())

        rank = env_vars["RANK"]
        local_rank = os.environ["LOCAL_RANK"]

        import torch
        torch.cuda.set_device(f"cuda:{local_rank}")

        print(
            f"[Rank {rank}] LOCAL_RANK={local_rank} "
            f"WORLD_SIZE={env_vars['WORLD_SIZE']} "
            f"MASTER_ADDR={env_vars['MASTER_ADDR']}:{env_vars['MASTER_PORT']}",
            flush=True,
        )

        sys.argv = ["sft_train.py", *sft_argv]

        from scripts.sft_train import main
        main()
        return f"[Rank {rank}] success"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Launch multi-node SFT training via Ray (one actor per GPU)",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=None,
        help="Number of nodes to use (default: all GPU nodes in the cluster)",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=8,
        help="Number of GPUs per node (default: 8)",
    )
    parser.add_argument(
        "--no-auto-convert-hf",
        action="store_true",
        default=False,
        help="Disable auto-converting checkpoints to HF format after training (enabled by default)",
    )
    args, sft_args = parser.parse_known_args()

    # Extract --model and --output-dir from sft_args for auto-convert
    sft_parser = argparse.ArgumentParser(add_help=False)
    sft_parser.add_argument("--model", type=str)
    sft_parser.add_argument("--output-dir", type=str)
    sft_known, _ = sft_parser.parse_known_args(sft_args)

    ray.init(address="auto", ignore_reinit_error=True)

    # Discover cluster topology
    available_nodes = [n for n in ray.nodes() if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0]
    if not available_nodes:
        raise SystemExit("No alive GPU nodes found in Ray cluster.")

    num_nodes = args.num_nodes if args.num_nodes is not None else len(available_nodes)
    if num_nodes > len(available_nodes):
        raise SystemExit(
            f"Requested {num_nodes} nodes but only {len(available_nodes)} GPU nodes available."
        )

    world_size = num_nodes * args.gpus_per_node
    print(f"[Ray SFT] {num_nodes} node(s), {args.gpus_per_node} GPUs/node, world_size={world_size}")
    print(f"[Ray SFT] sft_train.py args: {sft_args}")

    # Create placement group and discover topology
    pg, reordered_indices, node_ips = create_placement_group(world_size)

    # Figure out local_world_size per node and local_rank per rank
    # (consecutive ranks on same node thanks to reordering)
    node_rank_counts = Counter(node_ips)

    # Create actors, get master addr from rank 0
    actors = []
    master_addr, master_port = None, None
    for rank in range(world_size):
        actor = SFTWorker.options(
            num_gpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=reordered_indices[rank],
            ),
        ).remote()
        if rank == 0:
            master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            print(f"[Ray SFT] master={master_addr}:{master_port}")
        actors.append(actor)

    # Compute LOCAL_WORLD_SIZE for each rank
    # (all ranks on the same node share the same local_world_size)
    seen_nodes: dict[str, int] = {}  # node_ip -> count of ranks seen so far
    futures = []
    for rank, actor in enumerate(actors):
        node_ip = node_ips[rank]
        local_rank_val = seen_nodes.get(node_ip, 0)
        seen_nodes[node_ip] = local_rank_val + 1

        env_vars = {
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(node_rank_counts[node_ip]),
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }
        futures.append(actor.run.remote(sft_args, env_vars))

    results = ray.get(futures)
    for r in results:
        print(r, flush=True)
    print("[Ray SFT] All ranks finished successfully.", flush=True)

    # Kill all SFT worker actors to release GPU resources before conversion
    for actor in actors:
        ray.kill(actor)
    # Remove placement group to free reserved resources
    from ray.util.placement_group import remove_placement_group
    remove_placement_group(pg)
    import time
    time.sleep(5)  # Give Ray a moment to reclaim resources

    # Auto-convert checkpoints to HF format
    if not args.no_auto_convert_hf and sft_known.model and sft_known.output_dir:
        hf_model = sft_known.model
        ckpt_dir = sft_known.output_dir
        hf_output_dir = str(Path(ckpt_dir).parent / (Path(ckpt_dir).name + "-hf"))

        print(f"[Auto-Convert] Converting checkpoints to HF format...", flush=True)
        print(f"[Auto-Convert]   model:      {hf_model}", flush=True)
        print(f"[Auto-Convert]   input:      {ckpt_dir}", flush=True)
        print(f"[Auto-Convert]   output-dir: {hf_output_dir}", flush=True)

        # Convert each iter_* checkpoint in PARALLEL on its own actor (1 GPU each).
        # The convert_to_hf.py script uses `temporary_distributed_context(world_size=1)`
        # so it cannot multi-rank a single iter; the win is fanning iters out.
        # 397B with 3 ckpts went from ~2.6h serial to ~50min in parallel (3x).
        @ray.remote(num_cpus=4, num_gpus=1)
        def convert_one_iter(model_id, iter_path, out_dir):
            import subprocess, sys
            cmd = [
                sys.executable, "scripts/convert_to_hf.py",
                "--hf-model", model_id,
                "--input", iter_path,
                "--output-dir", out_dir,
            ]
            print(f"[Auto-Convert][{Path(iter_path).name}] Running", flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            tag = Path(iter_path).name
            if proc.returncode != 0:
                print(f"[Auto-Convert][{tag}] FAILED (exit {proc.returncode}):\n{proc.stderr}", flush=True)
            else:
                print(f"[Auto-Convert][{tag}] Done.", flush=True)
            return proc.returncode, tag

        # Discover iter dirs. ckpt_dir might be a bare output dir or contain a
        # nested checkpoints/ subdir (sft_train.py default). Match find_checkpoints'
        # discovery rule from convert_to_hf.py.
        iter_root = Path(ckpt_dir)
        if (iter_root / "checkpoints").is_dir():
            iter_root = iter_root / "checkpoints"
        iter_paths = sorted(p for p in iter_root.glob("iter_*") if p.is_dir())
        if not iter_paths:
            print(f"[Auto-Convert] No iter_* dirs under {iter_root}; nothing to convert.", flush=True)
        else:
            print(f"[Auto-Convert] Fanning out {len(iter_paths)} iter(s) in parallel:", flush=True)
            for p in iter_paths:
                print(f"[Auto-Convert]   - {p.name}", flush=True)
            refs = [convert_one_iter.remote(hf_model, str(p), hf_output_dir) for p in iter_paths]
            results = ray.get(refs)
            failed = [tag for code, tag in results if code != 0]
            if failed:
                print(f"[Auto-Convert] Conversion FAILED for: {failed}", flush=True)
            else:
                print(f"[Auto-Convert] All {len(results)} checkpoints converted to {hf_output_dir}", flush=True)
    elif not args.no_auto_convert_hf:
        print("[Auto-Convert] Skipped: --model or --output-dir not found in args", flush=True)


if __name__ == "__main__":
    main()
