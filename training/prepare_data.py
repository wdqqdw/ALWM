#!/usr/bin/env python3
"""Convert the public D1/D2/D3 metadata into training manifests, without relabeling."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

COUNTS = {"D1": 42673, "D2": 15290, "D3": 2885}
ANSWER = re.compile(r"<observation>(.*?)</observation>\n<reasoning>(.*?)</reasoning>\n<latent></latent>\n<prediction>(.*?)</prediction>(?:<answer>([ABCD])</answer>)?", re.S)
OPTIONS = re.compile(r"A\. (.*?)\nB\. (.*?)\nC\. (.*?)\nD\. (.*)", re.S)


def safe_video(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("video must be a nonempty repository-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("video path must stay inside the downloaded dataset")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("video symlink escapes the dataset")
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(root / path)
    return resolved


def duration(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)], check=True, capture_output=True, text=True, timeout=30)
    seconds = float(json.loads(result.stdout)["format"]["duration"])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"invalid video duration: {path}")
    return seconds


def parse_text(ds: str, sid: str, item: dict) -> dict:
    if set(item) != {"video", "query", "answer"}:
        raise ValueError(f"unexpected metadata schema: {ds}/{sid}")
    if not isinstance(item["query"], str) or not isinstance(item["answer"], str) or not item["answer"]:
        raise ValueError("query/answer must be text; answer cannot be empty")
    if ds == "D1":
        if item["query"] != "":
            raise ValueError("D1 query must be empty")
        return {"segment_id": sid, "video_relpath": item["video"], "caption": item["answer"], "question": "What is happening in the imagined scene?"}
    match = ANSWER.fullmatch(item["answer"])
    if match is None:
        raise ValueError(f"invalid tagged target: {ds}/{sid}")
    observation, reasoning, prediction, gold = match.groups()
    row = {"sample_id": sid, "question": item["query"], "video_understanding": observation, "future_reasoning": reasoning, "future_event_summary": prediction, "released_clips": True}
    if ds == "D2":
        if gold is not None or not isinstance(item["video"], dict) or set(item["video"]) != {"observation", "future"}:
            raise ValueError("D2 needs observation/future videos and no MC answer")
        row["video_relpath"] = item["video"]["observation"]
    else:
        question, separator, options_text = item["query"].rpartition("\n\nOptions:\n")
        options = OPTIONS.fullmatch(options_text)
        if not separator or not question or options is None or gold is None:
            raise ValueError("D3 needs a complete MC question and gold label")
        row.update(question=question, options=dict(zip("ABCD", options.groups())), answer=gold, video_relpath=item["video"])
        if row["options"][gold] != prediction:
            raise ValueError("D3 gold text differs from the labeled option")
    return row


def convert_row(root: Path, ds: str, sid: str, item: dict, *, probe: bool = True) -> dict:
    row = parse_text(ds, sid, item)
    paths = item["video"] if ds == "D2" else {"observation": item["video"]}
    intervals = {}
    for role, relative in paths.items():
        path = safe_video(root, relative)
        seconds = duration(path) if probe else None
        intervals[role] = {"video_relpath": relative, "start_seconds": 0.0, "end_seconds": seconds}
    if ds == "D1":
        # Clips are already trimmed. Do not seek or crop them a second time.
        row.update(start_seconds=0.0, end_seconds=intervals["observation"]["end_seconds"], duration_seconds=intervals["observation"]["end_seconds"], released_clips=True)
    else:
        row.update(context=intervals["observation"], future=intervals.get("future"))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--check-only", action="store_true", help="Validate all metadata/path mappings without probing or writing.")
    parser.add_argument("--limit", type=int, default=0, help="Explicit smoke-test subset per dataset; zero means all rows.")
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0:
        parser.error("workers must be positive and limit nonnegative")
    if not args.check_only:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    audit = {"counts": {}, "metadata_sha256": {}, "smoke_subset": bool(args.limit), "clip_timebase": "already_trimmed", "labels_modified": False}
    for ds, count in COUNTS.items():
        raw = (args.data_root / ds / f"{ds}_metadata.json").read_bytes()
        metadata = json.loads(raw)
        if not isinstance(metadata, dict) or len(metadata) != count:
            raise ValueError(f"{ds} must contain {count} sample IDs")
        items = list(metadata.items())[:args.limit or None]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(lambda pair: convert_row(args.data_root, ds, *pair, probe=not args.check_only), items))
        audit["counts"][ds] = len(rows)
        audit["metadata_sha256"][ds] = hashlib.sha256(raw).hexdigest()
        if not args.check_only:
            with (args.output_dir / f"{ds}.jsonl").open("x", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"dataset": ds, "checked": len(rows)}), flush=True)
    if not args.check_only:
        with (args.output_dir / "audit.json").open("x") as handle:
            json.dump(audit, handle, indent=2)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
