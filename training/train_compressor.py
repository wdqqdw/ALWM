#!/usr/bin/env python3
"""Train the eight-slot C1 compressor with a frozen Qwen3-VL backbone."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from alwm.compressors import build_compressor  # noqa: E402
from alwm.stage1 import (  # noqa: E402
    JsonlSceneDataset,
    attach_embedding_layer,
    build_caption_teacher_forcing_batch,
    encode_scene_tokens_batch,
    frozen_caption_losses,
    prepare_scene_model_inputs,
)
from alwm.training import (  # noqa: E402
    CostBucketDistributedSampler,
    JsonlMetricWriter,
    load_resume_checkpoint,
    save_resume_checkpoint,
    sha256_file,
    unwrap_module,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compressor", choices=["c1"], default="c1")
    parser.add_argument("--latent-tokens", type=int, default=8)
    parser.add_argument("--bottleneck-dim", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--video-prefetch-workers", type=int, default=1)
    parser.add_argument("--video-prefetch-depth", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--keep-last-checkpoints", type=int, default=1)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--duration-bucket-size", type=int, default=32)
    parser.add_argument("--attention-backend", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    if args.gradient_accumulation < 1 or args.log_every < 1:
        parser.error("gradient-accumulation and log-every must be positive")
    if args.micro_batch_size < 1 or args.gradient_accumulation % args.micro_batch_size:
        parser.error("micro-batch-size must be positive and divide gradient-accumulation")
    if args.checkpoint_every < 0:
        parser.error("checkpoint-every cannot be negative")
    if args.keep_last_checkpoints < 1:
        parser.error("keep-last-checkpoints must be positive")
    if args.video_prefetch_workers < 0 or args.video_prefetch_depth < 1:
        parser.error("video-prefetch-workers cannot be negative and depth must be positive")
    return args


def prefetch_scene_samples(
    indexed_samples,
    processor,
    fps: float,
    max_frames: int,
    workers: int,
    depth: int,
):
    """Overlap CPU video decoding with the current sample's GPU forward pass."""

    if workers == 0:
        yield from indexed_samples
        return

    def prepare(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        index, sample = item
        try:
            sample["_alwm_prepared_model_inputs"] = prepare_scene_model_inputs(
                sample, processor, fps, max_frames
            )
        except Exception as error:
            raise RuntimeError(
                "video prefetch failed for "
                f"segment_id={sample.get('segment_id')!r}, "
                f"video_relpath={sample.get('video_relpath')!r}, "
                f"interval=[{sample.get('start_seconds')!r}, {sample.get('end_seconds')!r}]"
            ) from error
        return index, sample

    queue: deque[Future] = deque()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="alwm-video") as executor:
        for item in indexed_samples:
            queue.append(executor.submit(prepare, item))
            if len(queue) >= depth:
                yield queue.popleft().result()
        while queue:
            yield queue.popleft().result()


def batch_prepared_samples(indexed_samples, batch_size: int):
    batch = []
    for item in indexed_samples:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        try:
            dist.init_process_group(backend="nccl", device_id=device)
        except TypeError:  # compatibility with older supported PyTorch builds
            dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, device


def load_backbone(model_path: str, device: torch.device, attention_backend: str):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attention_backend,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    attach_embedding_layer(processor.tokenizer, model)
    return model, processor


def model_hidden_size(model) -> int:
    text_config = getattr(model.config, "text_config", model.config)
    return int(text_config.hidden_size)


def reduce_sum(values: dict[str, float], device: torch.device, world_size: int) -> dict[str, float]:
    keys = sorted(values)
    tensor = torch.tensor([values[key] for key in keys], device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: float(tensor[index].item()) for index, key in enumerate(keys)}


def gather_memory(device: torch.device, world_size: int) -> list[list[float]]:
    gib = 1024**3
    local = torch.tensor(
        [
            torch.cuda.memory_allocated(device) / gib,
            torch.cuda.memory_reserved(device) / gib,
            torch.cuda.max_memory_allocated(device) / gib,
            torch.cuda.max_memory_reserved(device) / gib,
        ],
        device=device,
        dtype=torch.float64,
    )
    if world_size == 1:
        return [local.cpu().tolist()]
    gathered = [torch.zeros_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return [value.cpu().tolist() for value in gathered]


def gather_rank_metrics(
    values: dict[str, float], device: torch.device, world_size: int
) -> list[dict[str, float]]:
    """Collect an identically keyed metric dictionary from every rank."""

    keys = sorted(values)
    local = torch.tensor([values[key] for key in keys], device=device, dtype=torch.float64)
    if world_size == 1:
        tensors = [local]
    else:
        tensors = [torch.zeros_like(local) for _ in range(world_size)]
        dist.all_gather(tensors, local)
    return [
        {key: float(tensor[index].item()) for index, key in enumerate(keys)}
        for tensor in tensors
    ]


def checkpoint_state(epoch: int, batches_in_epoch: int, loader_length: int, global_step: int, micro_step: int) -> dict[str, Any]:
    next_epoch = epoch
    next_batch = batches_in_epoch
    if next_batch >= loader_length:
        next_epoch += 1
        next_batch = 0
    return {
        "schema_version": "alwm.stage1.resume.v1",
        "global_step": int(global_step),
        "micro_step_per_rank": int(micro_step),
        "next_epoch": int(next_epoch),
        "next_batch_in_epoch": int(next_batch),
    }


def save_final_checkpoint(
    output_dir: Path,
    compressor: torch.nn.Module,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    run_manifest: dict[str, Any],
) -> None:
    raw = unwrap_module(compressor)
    state = {name: tensor.detach().cpu() for name, tensor in raw.state_dict().items()}
    torch.save(state, output_dir / "compressor.pt")
    config = {
        "schema_version": "alwm.stage1.compressor.v2",
        "compressor": args.compressor,
        "latent_tokens": args.latent_tokens,
        "bottleneck_dim": args.bottleneck_dim,
        "backbone": args.model,
        "question": "What is happening in the imagined scene?",
        "fps": args.fps,
        "max_frames": args.max_frames,
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_accumulation": args.gradient_accumulation,
            "seed": args.seed,
            "world_size": run_manifest["world_size"],
            "effective_batch_size": run_manifest["effective_batch_size"],
        },
        "metrics": metrics,
    }
    write_json_atomic(output_dir / "config.json", config)
    write_json_atomic(output_dir / "summary.json", metrics)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen3-VL Stage-1 trainer")
    rank, local_rank, world_size, device = distributed_context()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
    if world_size > 1:
        dist.barrier()

    train_set = JsonlSceneDataset(args.train_manifest, args.video_root)
    if args.max_train_samples:
        train_set.rows = train_set.rows[: args.max_train_samples]
    if not len(train_set):
        raise ValueError("D1 manifest must be nonempty")
    workload_costs = [float(row["duration_seconds"]) for row in train_set.rows]
    sampler = None
    if world_size > 1:
        sampler = CostBucketDistributedSampler(
            workload_costs,
            num_replicas=world_size,
            rank=rank,
            bucket_size_per_rank=args.duration_bucket_size,
            seed=args.seed,
        )
    loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        collate_fn=lambda rows: rows[0],
    )

    model, processor = load_backbone(args.model, device, args.attention_backend)
    compressor = build_compressor(
        args.compressor,
        model_hidden_size(model),
        args.latent_tokens,
        args.bottleneck_dim,
    ).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in compressor.parameters())
    if world_size > 1:
        compressor = DistributedDataParallel(
            compressor,
            device_ids=[local_rank],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    optimizer = torch.optim.AdamW(
        compressor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    optimizer.zero_grad(set_to_none=True)

    run_manifest = {
        "schema_version": "alwm.stage1.run.v2",
        "compressor_code": args.compressor,
        "world_size": world_size,
        "effective_batch_size": world_size * args.gradient_accumulation,
        "train_samples": len(train_set),
        "trainable_parameters": trainable_parameters,
        "frozen_backbone_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_manifest": {"path": str(Path(args.train_manifest).resolve()), "sha256": sha256_file(args.train_manifest)},
        "args": vars(args),
    }
    if rank == 0:
        write_json_atomic(output_dir / "run_manifest.json", run_manifest)
    metric_writer = JsonlMetricWriter(output_dir / "metrics.jsonl") if rank == 0 else None

    start_epoch = 0
    skip_batches = 0
    global_step = 0
    micro_step = 0
    if args.resume_from:
        resume = load_resume_checkpoint(args.resume_from, compressor, optimizer, rank, device)
        start_epoch = int(resume["next_epoch"])
        skip_batches = int(resume["next_batch_in_epoch"])
        global_step = int(resume["global_step"])
        micro_step = int(resume["micro_step_per_rank"])

    running = {
        "loss_sum": 0.0,
        "samples": 0.0,
        "video_seconds": 0.0,
        "native_visual_tokens": 0.0,
        "target_tokens": 0.0,
        "encode_seconds": 0.0,
        "compressor_seconds": 0.0,
        "language_seconds": 0.0,
        "backward_seconds": 0.0,
        "optimizer_seconds": 0.0,
        "grad_norm_sum": 0.0,
        "optimizer_steps": 0.0,
    }
    training_started = time.perf_counter()
    window_started = training_started
    stop = False
    last_state = checkpoint_state(start_epoch, skip_batches, len(loader), global_step, micro_step)
    try:
        for epoch in range(start_epoch, args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            indexed_samples = (
                (batch_index, sample)
                for batch_index, sample in enumerate(loader)
                if not (epoch == start_epoch and batch_index < skip_batches)
            )
            prepared_samples = prefetch_scene_samples(
                indexed_samples,
                processor,
                args.fps,
                args.max_frames,
                args.video_prefetch_workers,
                args.video_prefetch_depth,
            )
            for prepared_batch in batch_prepared_samples(prepared_samples, args.micro_batch_size):
                batch_indices = [item[0] for item in prepared_batch]
                samples = [item[1] for item in prepared_batch]
                batch_index = batch_indices[-1]
                batch_size = len(samples)
                encode_started = time.perf_counter()
                features, times, mask, visual_metadata = encode_scene_tokens_batch(
                    model,
                    processor,
                    samples,
                    device,
                    args.fps,
                    args.max_frames,
                    return_metadata=True,
                )
                encode_elapsed = time.perf_counter() - encode_started
                will_step = (micro_step + batch_size) % args.gradient_accumulation == 0
                sync_context = nullcontext() if world_size == 1 or will_step else compressor.no_sync()
                with sync_context:
                    compressor_started = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        latent = compressor(features, times, mask)
                    compressor_elapsed = time.perf_counter() - compressor_started
                    language_started = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        embeds, attention, labels = build_caption_teacher_forcing_batch(
                            processor.tokenizer,
                            latent,
                            [sample["caption"] for sample in samples],
                            [sample["question"] for sample in samples],
                        )
                        loss = frozen_caption_losses(model, embeds, attention, labels).mean()
                        scaled_loss = loss * batch_size / args.gradient_accumulation
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite compressor loss")
                    language_elapsed = time.perf_counter() - language_started
                    backward_started = time.perf_counter()
                    scaled_loss.backward()
                    backward_elapsed = time.perf_counter() - backward_started
                micro_step += batch_size
                last_state = checkpoint_state(epoch, batch_index + 1, len(loader), global_step, micro_step)
                running["loss_sum"] += float(loss.detach().item()) * batch_size
                running["samples"] += float(batch_size)
                running["video_seconds"] += sum(
                    float(sample.get("duration_seconds", 0.0)) for sample in samples
                )
                running["native_visual_tokens"] += sum(
                    float(metadata["native_visual_tokens"]) for metadata in visual_metadata
                )
                running["target_tokens"] += float((labels[:, 1:] != -100).sum().item())
                running["encode_seconds"] += encode_elapsed
                running["compressor_seconds"] += compressor_elapsed
                running["language_seconds"] += language_elapsed
                running["backward_seconds"] += backward_elapsed
                if not will_step:
                    continue

                optimizer_started = time.perf_counter()
                grad_norm = torch.nn.utils.clip_grad_norm_(compressor.parameters(), args.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                running["optimizer_seconds"] += time.perf_counter() - optimizer_started
                running["grad_norm_sum"] += float(grad_norm.detach().item())
                running["optimizer_steps"] += 1.0
                global_step += 1
                last_state = checkpoint_state(epoch, batch_index + 1, len(loader), global_step, micro_step)

                if global_step % args.log_every == 0:
                    per_rank = gather_rank_metrics(running, device, world_size)
                    totals = reduce_sum(running, device, world_size)
                    memory = gather_memory(device, world_size)
                    wall = time.perf_counter() - window_started
                    payload: dict[str, Any] = {
                        "time": time.time(),
                        "train/global_step": global_step,
                        "train/epoch": epoch,
                        "train/epoch_fraction": (batch_index + 1) / max(len(loader), 1),
                        "train/loss": totals["loss_sum"] / max(totals["samples"], 1.0),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/grad_norm": totals["grad_norm_sum"] / max(totals["optimizer_steps"], 1.0),
                        "perf/samples_per_second_global": totals["samples"] / max(wall, 1e-6),
                        "perf/video_seconds_per_second_global": totals["video_seconds"] / max(wall, 1e-6),
                        "perf/native_visual_tokens_per_sample": totals["native_visual_tokens"] / max(totals["samples"], 1.0),
                        "perf/target_tokens_per_sample": totals["target_tokens"] / max(totals["samples"], 1.0),
                        "perf/encode_seconds_per_sample": totals["encode_seconds"] / max(totals["samples"], 1.0),
                        "perf/compressor_seconds_per_sample": totals["compressor_seconds"] / max(totals["samples"], 1.0),
                        "perf/language_seconds_per_sample": totals["language_seconds"] / max(totals["samples"], 1.0),
                        "perf/backward_seconds_per_sample": totals["backward_seconds"] / max(totals["samples"], 1.0),
                        "perf/optimizer_seconds_per_step": totals["optimizer_seconds"] / max(totals["optimizer_steps"], 1.0),
                        "perf/elapsed_seconds": time.perf_counter() - training_started,
                    }
                    for memory_rank, values in enumerate(memory):
                        payload[f"memory/rank_{memory_rank}_allocated_gib"] = values[0]
                        payload[f"memory/rank_{memory_rank}_reserved_gib"] = values[1]
                        payload[f"memory/rank_{memory_rank}_peak_allocated_gib"] = values[2]
                        payload[f"memory/rank_{memory_rank}_peak_reserved_gib"] = values[3]
                    for metric_rank, values in enumerate(per_rank):
                        rank_samples = max(values["samples"], 1.0)
                        payload[f"rank/rank_{metric_rank}_samples"] = values["samples"]
                        payload[f"rank/rank_{metric_rank}_video_seconds"] = values["video_seconds"]
                        payload[f"rank/rank_{metric_rank}_native_visual_tokens"] = values[
                            "native_visual_tokens"
                        ]
                        payload[f"rank/rank_{metric_rank}_native_visual_tokens_per_sample"] = (
                            values["native_visual_tokens"] / rank_samples
                        )
                        payload[f"rank/rank_{metric_rank}_encode_seconds"] = values[
                            "encode_seconds"
                        ]
                        payload[f"rank/rank_{metric_rank}_encode_seconds_per_sample"] = (
                            values["encode_seconds"] / rank_samples
                        )
                    if rank == 0:
                        metric_writer.write(payload)
                        write_json_atomic(output_dir / "progress.json", payload)
                        print(json.dumps(payload, separators=(",", ":")), flush=True)
                    for key in running:
                        running[key] = 0.0
                    torch.cuda.reset_peak_memory_stats(device)
                    window_started = time.perf_counter()

                if args.checkpoint_every and global_step % args.checkpoint_every == 0:
                    save_resume_checkpoint(
                        output_dir / "checkpoints",
                        compressor,
                        optimizer,
                        last_state,
                        rank,
                        world_size,
                        device,
                        keep_last=args.keep_last_checkpoints,
                    )
                if args.max_train_steps and global_step >= args.max_train_steps:
                    stop = True
                    break
            close_prefetch = getattr(prepared_samples, "close", None)
            if close_prefetch is not None:
                close_prefetch()
            skip_batches = 0
            if stop:
                break

        if micro_step % args.gradient_accumulation:
            if world_size > 1:
                for parameter in compressor.parameters():
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                        parameter.grad.div_(world_size)
            grad_norm = torch.nn.utils.clip_grad_norm_(compressor.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            last_state["global_step"] = global_step
            last_state["micro_step_per_rank"] = micro_step
            if rank == 0:
                print(json.dumps({"tail_optimizer_step": global_step, "grad_norm": float(grad_norm)}), flush=True)

        save_resume_checkpoint(
            output_dir / "checkpoints",
            compressor,
            optimizer,
            last_state,
            rank,
            world_size,
            device,
            keep_last=args.keep_last_checkpoints,
        )
        metrics = {
            "optimizer_steps": global_step,
            "micro_steps_per_rank": micro_step,
            "world_size": world_size,
            "effective_batch_size": world_size * args.gradient_accumulation,
            "training_elapsed_seconds": time.perf_counter() - training_started,
        }
        if rank == 0:
            save_final_checkpoint(output_dir, compressor, args, metrics, run_manifest)
            write_json_atomic(output_dir / "summary.json", metrics)
            print(json.dumps(metrics, indent=2), flush=True)
    finally:
        if metric_writer is not None:
            metric_writer.close()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
