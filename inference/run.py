#!/usr/bin/env python3
"""Run ALWM text -> eight continuous states -> text decoding on one observed clip."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import torch
from alwm.stage1 import encode_scene_tokens, multimodal_position_ids, _stop_token_ids
from alwm.loading import load_processor, load_inference_model


def select_input(args):
    if args.video:
        if not args.question or args.sample_id:
            raise ValueError("--video needs --question and cannot be combined with --sample-id")
        return str(Path(args.video).resolve()), args.question, None
    from prepare_data import safe_video
    root = Path(args.data_root).resolve()
    rows = json.loads((root / "D3/D3_metadata.json").read_text())
    sid = args.sample_id or next(iter(rows))
    item = rows[sid]
    # Deliberately do not read answer, target reasoning, or a future video.
    return str(safe_video(root, item["video"])), item["query"], sid


@torch.inference_mode()
def decode(model, tokenizer, visual, query, *, reasoning_budget=512, prediction_budget=192):
    sentinel = "<<ALWM_OBSERVED_VIDEO_TOKENS>>"
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": sentinel + "\n" + query}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if rendered.count(sentinel) != 1:
        raise ValueError("video sentinel must appear exactly once")
    before, after = rendered.split(sentinel)
    device = visual.device
    embed = model.get_input_embeddings()
    ids = lambda text: torch.tensor(tokenizer.encode(text, add_special_tokens=False), device=device, dtype=torch.long)
    inputs = torch.cat([embed(ids(before)), visual[0], embed(ids(after))], dim=0)[None]
    attention = torch.ones(inputs.shape[:2], device=device, dtype=torch.long)
    language = model.model.language_model
    output = language(inputs_embeds=inputs, attention_mask=attention, position_ids=multimodal_position_ids(1, inputs.shape[1], device), use_cache=True, visual_pos_masks=None, deepstack_visual_embeds=None, return_dict=True)
    past = output.past_key_values
    position = inputs.shape[1]

    def consume(value):
        nonlocal attention, past, position
        attention = torch.cat([attention, attention.new_ones((1, 1))], dim=1)
        result = language(inputs_embeds=value, attention_mask=attention, position_ids=multimodal_position_ids(1, 1, device, position), past_key_values=past, use_cache=True, visual_pos_masks=None, deepstack_visual_embeds=None, return_dict=True)
        past = result.past_key_values
        position += 1
        return result

    stops = _stop_token_ids(tokenizer)
    pre, post = [], []
    marker = False
    for _ in range(reasoning_budget):
        token = model.lm_head(output.last_hidden_state[:, -1]).argmax(dim=-1)
        value = int(token.item())
        pre.append(value)
        output = consume(embed(token[:, None]))
        if "<latent>" in tokenizer.decode(pre, skip_special_tokens=False):
            marker = True
            break
        if value in stops:
            break
    if not marker:
        return {"text": tokenizer.decode(pre, skip_special_tokens=False), "complete": False, "latent_count": 0, "stop_reason": "latent_marker_not_generated"}
    # The hidden state after <latent> is z1. Feed z1...z7 to obtain z2...z8,
    # then feed z8 before sampling the first post-latent text token. Same KV.
    latents = [output.last_hidden_state[:, -1:]]
    for _ in range(7):
        output = consume(latents[-1])
        latents.append(output.last_hidden_state[:, -1:])
    output = consume(latents[-1])
    closed = False
    for _ in range(prediction_budget):
        token = model.lm_head(output.last_hidden_state[:, -1]).argmax(dim=-1)
        value = int(token.item())
        post.append(value)
        if value in stops:
            break
        output = consume(embed(token[:, None]))
        text = tokenizer.decode(post, skip_special_tokens=False)
        closed = "</prediction>" in text and ("Options:\n" not in query or "</answer>" in text)
        if closed:
            break
    latent_tensor = torch.cat(latents, dim=1)
    finite = bool(torch.isfinite(latent_tensor).all())
    return {"text": tokenizer.decode(pre + post, skip_special_tokens=False).strip(), "complete": closed and finite, "latent_count": 8, "latent_shape": list(latent_tensor.shape), "latent_finite": finite, "reasoning_tokens": len(pre), "prediction_tokens": len(post), "stop_reason": "complete" if closed else "incomplete_prediction", "kv_reset": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local full HF checkpoint directory")
    parser.add_argument("--sft-checkpoint", help="Optional local P3 checkpoint containing language_module.pt; use with the base model")
    parser.add_argument("--data-root", default="data/ALWM_data")
    parser.add_argument("--sample-id")
    parser.add_argument("--video")
    parser.add_argument("--question")
    parser.add_argument("--output", default="outputs/inference/result.json")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--reasoning-budget", type=int, default=512)
    parser.add_argument("--prediction-budget", type=int, default=192)
    args = parser.parse_args()
    if args.fps <= 0 or min(args.max_frames, args.reasoning_budget, args.prediction_budget) < 1:
        parser.error("frame and token budgets must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"preserving existing output: {output}")
    video, query, sid = select_input(args)
    if not Path(video).is_file():
        raise FileNotFoundError(video)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    started = time.perf_counter()
    processor = load_processor(args.model)
    model = load_inference_model(args.model, torch.bfloat16).to("cuda").eval()
    if args.sft_checkpoint:
        state = torch.load(Path(args.sft_checkpoint) / "language_module.pt", map_location="cpu", weights_only=True)
        state = {key.removeprefix("qwen."): value for key, value in state.items()}
        expected = {key for key in model.state_dict() if key.startswith(("model.language_model.", "lm_head."))}
        if set(state) != expected:
            raise ValueError("SFT checkpoint must contain the complete language model and LM head")
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise ValueError(result.unexpected_keys)
    model.requires_grad_(False)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        features, _, _, metadata = encode_scene_tokens(model, processor, {"video_path": video, "start_seconds": None, "end_seconds": None}, torch.device("cuda"), args.fps, args.max_frames, return_metadata=True)
        result = decode(model, processor.tokenizer, features, query, reasoning_budget=args.reasoning_budget, prediction_budget=args.prediction_budget)
    result.update(sample_id=sid, video=video, question=query, visual=metadata, elapsed_seconds=time.perf_counter() - started, peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3, input_video="observed_only", target_answer_loaded=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
