#!/usr/bin/env python3
"""Train P1/P2/P3 SFT with a frozen visual encoder and frozen C1 teacher."""

from __future__ import annotations

import argparse
import json
import math
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from alwm.compressors import build_compressor  # noqa: E402
from alwm.stage1 import (  # noqa: E402
    attach_embedding_layer,
    encode_scene_tokens_batch,
    prepare_scene_model_inputs,
)
from alwm.stage2 import CurriculumDataset, FutureLatentTrainingModule  # noqa: E402
from alwm.training import (  # noqa: E402
    JsonlMetricWriter,
    sha256_file,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ab-train-manifest", required=True)
    parser.add_argument("--c-train-manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--compressor-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--a-end-fraction", type=float, default=0.25)
    parser.add_argument("--b-end-fraction", type=float, default=0.85)
    parser.add_argument("--a-end-step", type=int, default=0)
    parser.add_argument("--b-end-step", type=int, default=0)
    parser.add_argument(
        "--stop-after-step",
        type=int,
        default=0,
        help="Stop and save a formal phase-boundary checkpoint before total_steps.",
    )
    parser.add_argument("--c-b-rehearsal-ratio", type=float, default=0.20)
    parser.add_argument(
        "--c-local-data-indexing",
        action="store_true",
        help=(
            "Index phase-C rows with a resume-stable C-only cursor. This keeps "
            "periodic B rehearsal from skipping C rows and enables exact coverage audits."
        ),
    )
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--c-learning-rate",
        type=float,
        default=0.0,
        help=(
            "Optional phase-C learning-rate restart. Zero preserves the original global "
            "schedule; a positive value starts a C-local warmup/cosine schedule."
        ),
    )
    parser.add_argument(
        "--c-warmup-ratio",
        type=float,
        default=0.05,
        help="Warmup fraction within phase C when --c-learning-rate is positive.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--alignment-weight", type=float, default=1.0)
    parser.add_argument("--caption-weight", type=float, default=1.0)
    parser.add_argument("--post-text-weight", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--video-prefetch-workers", type=int, default=8)
    parser.add_argument("--video-prefetch-depth", type=int, default=8)
    parser.add_argument(
        "--duration-bucketing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pair similar estimated frame counts across DDP ranks to reduce stragglers.",
    )
    parser.add_argument("--attention-backend", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--optimizer", choices=["adamw", "adamw-fused", "adamw8bit"], default="adamw8bit")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume-from", default="")
    args = parser.parse_args()
    if not 0 < args.a_end_fraction < args.b_end_fraction < 1:
        parser.error("require 0 < a-end-fraction < b-end-fraction < 1")
    if not 0 <= args.c_b_rehearsal_ratio < 1:
        parser.error("c-b-rehearsal-ratio must be in [0,1)")
    if args.c_learning_rate < 0:
        parser.error("c-learning-rate must be non-negative")
    if not 0 <= args.c_warmup_ratio < 1:
        parser.error("c-warmup-ratio must be in [0,1)")
    if args.total_steps < 1 or args.gradient_accumulation < 1:
        parser.error("total-steps and gradient-accumulation must be positive")
    if args.log_every < 1 or args.checkpoint_every < 0:
        parser.error("log-every must be positive and checkpoint-every nonnegative")
    if args.a_end_step or args.b_end_step:
        if not 0 < args.a_end_step < args.b_end_step < args.total_steps:
            parser.error("explicit boundaries require 0 < a-end-step < b-end-step < total")
    if args.stop_after_step and not 0 < args.stop_after_step <= args.total_steps:
        parser.error("stop-after-step must be in (0,total-steps]")
    if args.video_prefetch_workers < 0 or args.video_prefetch_depth < 1:
        parser.error("video-prefetch-workers cannot be negative and depth must be positive")
    return args


def distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        try:
            dist.init_process_group("nccl", device_id=device)
        except TypeError:
            dist.init_process_group("nccl")
    return rank, local_rank, world_size, device


def phase_boundaries(args: argparse.Namespace) -> tuple[int, int]:
    a_end_step = int(getattr(args, "a_end_step", 0))
    b_end_step = int(getattr(args, "b_end_step", 0))
    if a_end_step and b_end_step:
        return a_end_step, b_end_step
    return (
        round(args.total_steps * getattr(args, "a_end_fraction", 0.25)),
        round(args.total_steps * getattr(args, "b_end_fraction", 0.85)),
    )


def phase_for_args(step: int, args: argparse.Namespace) -> str:
    a_end, b_end = phase_boundaries(args)
    if step < a_end:
        return "a"
    if step < b_end:
        return "b"
    return "c"


def scheduled_learning_rate(step: int, args: argparse.Namespace) -> float:
    _, c_start = phase_boundaries(args)
    if args.c_learning_rate > 0 and step >= c_start:
        c_steps = max(args.total_steps - c_start, 1)
        c_local_step = step - c_start
        c_warmup = max(1, round(c_steps * args.c_warmup_ratio))
        if c_local_step < c_warmup:
            return args.c_learning_rate * (c_local_step + 1) / c_warmup
        progress = (c_local_step - c_warmup) / max(c_steps - c_warmup, 1)
        return args.c_learning_rate * 0.5 * (
            1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
        )
    warmup = max(1, round(args.total_steps * args.warmup_ratio))
    if step < warmup:
        return args.learning_rate * (step + 1) / warmup
    progress = (step - warmup) / max(args.total_steps - warmup, 1)
    return args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def load_model(args: argparse.Namespace, device: torch.device):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attention_backend,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.model.language_model.parameters():
        parameter.requires_grad_(True)
    for parameter in model.lm_head.parameters():
        parameter.requires_grad_(True)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        enable = getattr(model.model.language_model, "gradient_checkpointing_enable", None)
        if enable is not None:
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.eval()
    model.model.language_model.train()
    model.lm_head.train()
    attach_embedding_layer(processor.tokenizer, model)
    return model, processor


def load_compressor(checkpoint: str | Path, hidden_size: int, device: torch.device):
    directory = Path(checkpoint)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    compressor = build_compressor(
        config["compressor"],
        hidden_size,
        int(config["latent_tokens"]),
        int(config["bottleneck_dim"]),
    ).to(device=device, dtype=torch.bfloat16)
    compressor.load_state_dict(
        torch.load(directory / "compressor.pt", map_location=device, weights_only=True)
    )
    compressor.eval()
    for parameter in compressor.parameters():
        parameter.requires_grad_(False)
    return compressor, config


def save_checkpoint(
    root: Path,
    module: FutureLatentTrainingModule | DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    rank: int,
    world_size: int,
) -> Path:
    directory = root / f"step-{state['global_step']:06d}"
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    torch.save(
        {
            "rank": rank,
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "python_rng": random.getstate(),
        },
        directory / f"rank-{rank:05d}.pt",
    )
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        raw = module.module if isinstance(module, DistributedDataParallel) else module
        trainable_names = {
            name for name, parameter in raw.named_parameters() if parameter.requires_grad
        }
        trainable = {
            name: value.detach().cpu()
            for name, value in raw.state_dict().items()
            if name in trainable_names
        }
        torch.save(trainable, directory / "language_module.pt")
        torch.save({"optimizer": optimizer.state_dict(), "state": state}, directory / "trainer.pt")
        write_json_atomic(directory / "state.json", state)
        for stale in sorted(root.glob("step-*"), key=lambda path: int(path.name[5:]))[:-1]:
            import shutil

            shutil.rmtree(stale)
        (root / "latest.txt").write_text(str(directory.resolve()) + "\n", encoding="utf-8")
    if world_size > 1:
        dist.barrier()
    return directory


def load_checkpoint(
    directory: Path,
    module: FutureLatentTrainingModule | DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    raw = module.module if isinstance(module, DistributedDataParallel) else module
    weights = torch.load(directory / "language_module.pt", map_location=device, weights_only=True)
    expected = {name for name, value in raw.named_parameters() if value.requires_grad}
    if set(weights) != expected:
        raise ValueError("checkpoint does not contain exactly the trainable language parameters")
    raw.load_state_dict(weights, strict=False)
    payload = torch.load(directory / "trainer.pt", map_location=device, weights_only=False)
    optimizer.load_state_dict(payload["optimizer"])
    rank = int(os.environ.get("RANK", "0"))
    rng = torch.load(
        directory / f"rank-{rank:05d}.pt", map_location="cpu", weights_only=False
    )
    torch.set_rng_state(rng["torch_cpu_rng"])
    if device.type == "cuda" and rng["torch_cuda_rng"] is not None:
        torch.cuda.set_rng_state(rng["torch_cuda_rng"], device)
    random.setstate(rng["python_rng"])
    return dict(payload["state"])


def deterministic_index(
    micro_step: int, rank: int, world_size: int, length: int, seed: int
) -> int:
    global_sample = micro_step * world_size + rank
    cycle, offset = divmod(global_sample, length)
    generator = random.Random(seed + cycle)
    permutation = list(range(length))
    generator.shuffle(permutation)
    return permutation[offset]


def estimated_frame_cost(row: dict[str, Any], fps: float, max_frames: int) -> float:
    """Cheap decode/vision-cost proxy available before opening the video."""

    total = 0.0
    for key in ("context", "future"):
        interval = row.get(key)
        if interval is None:
            continue
        duration = max(0.0, float(interval["end_seconds"]) - float(interval["start_seconds"]))
        total += min(float(max_frames), max(1.0, duration * fps))
    return total


def duration_order(dataset: CurriculumDataset, fps: float, max_frames: int) -> list[int]:
    return sorted(
        range(len(dataset)),
        key=lambda index: (
            estimated_frame_cost(dataset.rows[index], fps, max_frames),
            str(dataset.rows[index].get("sample_id", index)),
        ),
    )


def deterministic_bucketed_index(
    micro_step: int,
    rank: int,
    world_size: int,
    ordered_indices: list[int],
    seed: int,
) -> int:
    """Give every DDP rank a similarly sized sample without losing coverage."""

    if not ordered_indices:
        raise ValueError("ordered_indices cannot be empty")
    bucket_count = math.ceil(len(ordered_indices) / world_size)
    cycle, bucket_offset = divmod(micro_step, bucket_count)
    generator = random.Random(seed + cycle)
    bucket_permutation = list(range(bucket_count))
    generator.shuffle(bucket_permutation)
    bucket_index = bucket_permutation[bucket_offset]
    bucket = ordered_indices[bucket_index * world_size : (bucket_index + 1) * world_size]
    # Rotate rank assignment each epoch.  Only the final incomplete bucket can
    # repeat a sample, at most world_size-1 examples per dataset traversal.
    slot = (rank + cycle) % len(bucket)
    return bucket[slot]


def select_training_work(
    args: argparse.Namespace,
    ab_set: CurriculumDataset,
    c_set: CurriculumDataset,
    micro_step: int,
    global_step: int,
    rank: int,
    world_size: int,
    ab_duration_order: list[int] | None = None,
    c_duration_order: list[int] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    nominal_phase = phase_for_args(global_step, args)
    phase = nominal_phase
    dataset = ab_set
    if nominal_phase == "c":
        rehearsal_stride = max(1, round(1.0 / max(args.c_b_rehearsal_ratio, 1e-9)))
        if args.c_b_rehearsal_ratio > 0 and micro_step % rehearsal_stride == 0:
            phase = "b"
        else:
            dataset = c_set
    index_step = micro_step
    if dataset is c_set and args.c_local_data_indexing:
        _, c_start_global_step = phase_boundaries(args)
        c_start_micro_step = c_start_global_step * args.gradient_accumulation
        elapsed_c_micro_steps = max(0, micro_step - c_start_micro_step)
        rehearsal_stride = max(1, round(1.0 / max(args.c_b_rehearsal_ratio, 1e-9)))
        rehearsal_steps = 0
        if args.c_b_rehearsal_ratio > 0 and elapsed_c_micro_steps:
            # Count stride-aligned rehearsal micro-steps in
            # [c_start_micro_step, micro_step). The resulting cursor is stable
            # across prefetching and checkpoint resume without extra state.
            rehearsal_steps = (micro_step - 1) // rehearsal_stride - (
                c_start_micro_step - 1
            ) // rehearsal_stride
        index_step = elapsed_c_micro_steps - rehearsal_steps
    order = c_duration_order if dataset is c_set else ab_duration_order
    if args.duration_bucketing and order is not None and world_size > 1:
        index = deterministic_bucketed_index(index_step, rank, world_size, order, args.seed)
    else:
        index = deterministic_index(index_step, rank, world_size, len(dataset), args.seed)
    return nominal_phase, phase, dataset[index]


def prepare_training_row(
    work: tuple[str, str, dict[str, Any]],
    processor: Any,
    fps: float,
    max_frames: int,
) -> tuple[str, str, dict[str, Any]]:
    nominal_phase, phase, row = work
    for key in ("context_scene", "future_scene"):
        if key not in row or (phase == "c" and key == "future_scene"):
            continue
        row[key]["_alwm_prepared_model_inputs"] = prepare_scene_model_inputs(
            row[key], processor, fps, max_frames
        )
    return nominal_phase, phase, row


class OrderedTrainingPrefetcher:
    """Decode future context/future pairs while the current pair runs on GPU."""

    def __init__(
        self,
        args: argparse.Namespace,
        ab_set: CurriculumDataset,
        c_set: CurriculumDataset,
        processor: Any,
        rank: int,
        world_size: int,
        start_micro_step: int,
        start_global_step: int,
    ):
        self.args = args
        self.ab_set = ab_set
        self.c_set = c_set
        self.processor = processor
        self.rank = rank
        self.world_size = world_size
        self.start_micro_step = start_micro_step
        self.start_global_step = start_global_step
        self.ab_duration_order = duration_order(ab_set, args.fps, args.max_frames)
        self.c_duration_order = duration_order(c_set, args.fps, args.max_frames)
        self.executor = (
            ThreadPoolExecutor(
                max_workers=args.video_prefetch_workers,
                thread_name_prefix=f"stage2-video-rank{rank}",
            )
            if args.video_prefetch_workers
            else None
        )
        self.queue_span = args.video_prefetch_depth if self.executor else 1
        self.queue: deque[tuple[int, Future | tuple[str, str, dict[str, Any]]]] = deque()
        for offset in range(self.queue_span):
            self._submit(start_micro_step + offset)

    def _projected_global_step(self, candidate_micro_step: int) -> int:
        completed = candidate_micro_step - self.start_micro_step
        initial_remainder = self.start_micro_step % self.args.gradient_accumulation
        return self.start_global_step + (
            initial_remainder + completed
        ) // self.args.gradient_accumulation

    def _submit(self, candidate_micro_step: int) -> None:
        work = select_training_work(
            self.args,
            self.ab_set,
            self.c_set,
            candidate_micro_step,
            self._projected_global_step(candidate_micro_step),
            self.rank,
            self.world_size,
            self.ab_duration_order,
            self.c_duration_order,
        )
        if self.executor is None:
            value: Future | tuple[str, str, dict[str, Any]] = work
        else:
            value = self.executor.submit(
                prepare_training_row,
                work,
                self.processor,
                self.args.fps,
                self.args.max_frames,
            )
        self.queue.append((candidate_micro_step, value))

    def pop(self, expected_micro_step: int) -> tuple[str, str, dict[str, Any]]:
        candidate, value = self.queue.popleft()
        if candidate != expected_micro_step:
            raise RuntimeError("training prefetch queue lost deterministic ordering")
        result = value.result() if isinstance(value, Future) else value
        self._submit(expected_micro_step + self.queue_span)
        return result

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = distributed_context()
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    output_dir = Path(args.output_dir).resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "checkpoints").mkdir()
    if world_size > 1:
        dist.barrier()

    model, processor = load_model(args, device)
    text_config = getattr(model.config, "text_config", model.config)
    compressor, compressor_config = load_compressor(
        args.compressor_checkpoint, int(text_config.hidden_size), device
    )
    ab_set = CurriculumDataset(args.ab_train_manifest, args.video_root)
    c_set = CurriculumDataset(args.c_train_manifest, args.video_root)
    if not len(ab_set) or not len(c_set):
        raise ValueError("D2 and D3 manifests must be nonempty")
    train_module = FutureLatentTrainingModule(
        model, processor.tokenizer, args.alignment_weight,
        args.caption_weight, args.post_text_weight,
    )
    if world_size > 1:
        train_module = DistributedDataParallel(
            train_module, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    parameters = [p for p in train_module.parameters() if p.requires_grad]
    if args.optimizer == "adamw8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            parameters, lr=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            parameters, lr=args.learning_rate, weight_decay=args.weight_decay,
            fused=args.optimizer == "adamw-fused",
        )
    if rank == 0:
        write_json_atomic(output_dir / "run_manifest.json", {
            "world_size": world_size,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "frozen_compressor": compressor_config,
            "manifest_sha256": {
                "D2": sha256_file(args.ab_train_manifest),
                "D3": sha256_file(args.c_train_manifest),
            },
            "args": vars(args),
        })
    metric_writer = JsonlMetricWriter(output_dir / "metrics.jsonl") if rank == 0 else None
    global_step, micro_step = 0, 0
    if args.resume_from:
        state = load_checkpoint(Path(args.resume_from), train_module, optimizer, device)
        global_step, micro_step = int(state["global_step"]), int(state["micro_step"])
    target_step = args.stop_after_step or args.total_steps
    if global_step >= target_step:
        raise ValueError("resume step must be before the requested stop step")
    if micro_step != global_step * args.gradient_accumulation:
        raise ValueError("resume must be an optimizer boundary with unchanged accumulation")
    prefetcher = OrderedTrainingPrefetcher(
        args, ab_set, c_set, processor, rank, world_size, micro_step, global_step
    )
    optimizer.zero_grad(set_to_none=True)
    running_loss, running_samples = 0.0, 0
    started = time.perf_counter()
    try:
        while global_step < target_step:
            nominal_phase, phase, row = prefetcher.pop(micro_step)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                scenes = [row["context_scene"]]
                if phase != "c":
                    scenes.append(row["future_scene"])
                features, times, masks = encode_scene_tokens_batch(
                    model, processor, scenes, device, args.fps, args.max_frames
                )
                observed = features[:1, masks[0]].detach()
                teacher = None if phase == "c" else compressor(
                    features[1:], times[1:], masks[1:]
                ).detach()
            will_step = (micro_step + 1) % args.gradient_accumulation == 0
            sync = nullcontext() if world_size == 1 or will_step else train_module.no_sync()
            # no_sync encloses both forward and backward for DDP accumulation.
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    results = train_module(phase, observed, teacher, row)
                    loss = results["loss"] / args.gradient_accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite SFT loss")
                loss.backward()
            micro_step += 1
            running_loss += float(results["loss"].detach())
            running_samples += 1
            if not will_step:
                continue
            learning_rate = scheduled_learning_rate(global_step, args)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, args.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % args.log_every == 0 or global_step == target_step:
                values = torch.tensor([running_loss, running_samples], device=device, dtype=torch.float64)
                if world_size > 1:
                    dist.all_reduce(values)
                if rank == 0:
                    payload = {
                        "global_step": global_step, "micro_step": micro_step,
                        "phase": nominal_phase, "loss": float(values[0] / values[1]),
                        "learning_rate": learning_rate, "grad_norm": float(grad_norm),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                    metric_writer.write(payload)
                    write_json_atomic(output_dir / "progress.json", payload)
                    print(json.dumps(payload), flush=True)
                running_loss, running_samples = 0.0, 0
            if args.checkpoint_every and global_step % args.checkpoint_every == 0:
                save_checkpoint(output_dir / "checkpoints", train_module, optimizer,
                    {"global_step": global_step, "micro_step": micro_step}, rank, world_size)
        checkpoint = save_checkpoint(output_dir / "checkpoints", train_module, optimizer,
            {"global_step": global_step, "micro_step": micro_step}, rank, world_size)
        if rank == 0:
            summary = {
                "status": "training_complete" if target_step == args.total_steps else "phase_boundary_complete",
                "global_step": global_step, "micro_step": micro_step,
                "last_checkpoint": str(checkpoint),
            }
            write_json_atomic(output_dir / "summary.json", summary)
            print(json.dumps(summary), flush=True)
    finally:
        prefetcher.close()
        if metric_writer:
            metric_writer.close()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
