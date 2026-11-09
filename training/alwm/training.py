"""Training logs, deterministic sampling, and local checkpoint utilities."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


def sha256_file(path: str | Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


class JsonlMetricWriter:
    """Append-only, line-buffered local metric log."""

    def __init__(self, path: str | Path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.handle = target.open("a", encoding="utf-8", buffering=1)

    def write(self, payload: dict[str, Any]) -> None:
        self.handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class DurationBucketDistributedSampler(Sampler[int]):
    """Shard similarly sized video intervals across ranks for each DDP step.

    The complete index set is sorted by annotated interval duration, divided
    into local-duration buckets, and shuffled deterministically per epoch.
    Consecutive entries therefore have similar visual workloads before the
    list is strided across ranks. A final incomplete bucket is padded with
    repeated indices so all ranks perform the same number of updates.
    """

    def __init__(
        self,
        durations: Sequence[float],
        num_replicas: int,
        rank: int,
        bucket_size_per_rank: int = 32,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed sampler rank/world size")
        if bucket_size_per_rank < 1:
            raise ValueError("bucket_size_per_rank must be positive")
        self.durations = [float(value) for value in durations]
        self.num_replicas = num_replicas
        self.rank = rank
        self.bucket_size_per_rank = bucket_size_per_rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        if drop_last:
            self.num_samples = len(self.durations) // num_replicas
        else:
            self.num_samples = (len(self.durations) + num_replicas - 1) // num_replicas
        self.total_size = self.num_samples * num_replicas

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        ordered = sorted(range(len(self.durations)), key=self.durations.__getitem__)
        width = self.bucket_size_per_rank * self.num_replicas
        buckets = [ordered[start : start + width] for start in range(0, len(ordered), width)]
        bucket_order = torch.randperm(len(buckets), generator=generator).tolist()
        shuffled: list[int] = []
        for bucket_index in bucket_order:
            bucket = buckets[bucket_index]
            permutation = torch.randperm(len(bucket), generator=generator).tolist()
            shuffled_bucket = [bucket[index] for index in permutation]
            remainder = len(shuffled_bucket) % self.num_replicas
            if remainder:
                if self.drop_last:
                    shuffled_bucket = shuffled_bucket[: len(shuffled_bucket) - remainder]
                else:
                    padding = self.num_replicas - remainder
                    shuffled_bucket.extend((shuffled_bucket * self.num_replicas)[:padding])
            shuffled.extend(shuffled_bucket)
        if self.drop_last:
            shuffled = shuffled[: self.total_size]
        elif len(shuffled) != self.total_size:
            raise RuntimeError(
                f"duration sampler expected {self.total_size} global indices, produced {len(shuffled)}"
            )
        indices = shuffled[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("duration sampler produced an imbalanced shard")
        return iter(indices)


class CostBucketDistributedSampler(DurationBucketDistributedSampler):
    """Semantic alias for bucketing by any estimated per-sample workload."""


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": __import__("random").getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    __import__("random").setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def prune_resume_checkpoints(checkpoint_root: str | Path, keep_last: int = 1) -> list[Path]:
    """Keep only the newest completed ``step-*`` checkpoint directories."""

    if keep_last < 1:
        raise ValueError("keep_last must be positive")
    root = Path(checkpoint_root)
    checkpoints = sorted(
        (
            path
            for path in root.glob("step-*")
            if path.is_dir() and path.name.startswith("step-") and path.name[5:].isdigit()
        ),
        key=lambda path: int(path.name[5:]),
    )
    for stale in checkpoints[:-keep_last]:
        shutil.rmtree(stale)
    return checkpoints[-keep_last:]


def save_resume_checkpoint(
    checkpoint_root: str | Path,
    compressor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    rank: int,
    world_size: int,
    device: torch.device,
    keep_last: int = 1,
) -> Path:
    directory = Path(checkpoint_root) / f"step-{int(state['global_step']):06d}"
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    torch.save(
        {"rank": rank, "rng": capture_rng_state(device)},
        directory / f"rank-{rank:05d}.pt",
    )
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        raw = unwrap_module(compressor)
        torch.save({name: value.detach().cpu() for name, value in raw.state_dict().items()}, directory / "compressor.pt")
        torch.save({"optimizer": optimizer.state_dict(), "state": state}, directory / "trainer.pt")
        write_json_atomic(directory / "state.json", state)
        latest = Path(checkpoint_root) / "latest.txt"
        latest_temporary = latest.with_suffix(".txt.tmp")
        latest_temporary.write_text(str(directory.resolve()) + "\n", encoding="utf-8")
        latest_temporary.replace(latest)
        prune_resume_checkpoints(checkpoint_root, keep_last=keep_last)
    if world_size > 1:
        dist.barrier()
    return directory


def load_resume_checkpoint(
    checkpoint: str | Path,
    compressor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rank: int,
    device: torch.device,
) -> dict[str, Any]:
    directory = Path(checkpoint)
    raw = unwrap_module(compressor)
    raw.load_state_dict(torch.load(directory / "compressor.pt", map_location=device, weights_only=True))
    trainer = torch.load(directory / "trainer.pt", map_location=device, weights_only=False)
    optimizer.load_state_dict(trainer["optimizer"])
    for values in optimizer.state.values():
        for key, value in values.items():
            if torch.is_tensor(value):
                values[key] = value.to(device)
    rank_state = torch.load(directory / f"rank-{rank:05d}.pt", map_location="cpu", weights_only=False)
    restore_rng_state(rank_state["rng"], device)
    return dict(trainer["state"])
