"""Shared Stage-1 data, visual encoding, latent injection, and decoding code."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


LATENT_SENTINEL = "<<ALWM_LATENT_SLOTS>>"
SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_QUESTION = "What is happening in the imagined scene?"
_VIDEO_READER_PATCH_LOCK = threading.Lock()


def install_strict_torchcodec_reader(vision_process: Any) -> None:
    """Retry TorchCodec end-of-stream edges and prohibit silent slow fallback."""

    if getattr(vision_process, "_alwm_strict_torchcodec_installed", False):
        return
    with _VIDEO_READER_PATCH_LOCK:
        if getattr(vision_process, "_alwm_strict_torchcodec_installed", False):
            return
        if vision_process.get_video_reader_backend() != "torchcodec":
            return
        original = vision_process.VIDEO_READER_BACKENDS["torchcodec"]

        def resilient_torchcodec(element: dict[str, Any]):
            try:
                return original(element)
            except Exception as error:
                message = str(error).lower()
                end = element.get("video_end")
                start = float(element.get("video_start", 0.0))
                if end is None or not any(
                    marker in message
                    for marker in ("no more frames", "out of range", "past the end")
                ):
                    raise
                for trim_seconds in (0.05, 0.10, 0.25, 0.50, 1.00):
                    adjusted_end = float(end) - trim_seconds
                    if adjusted_end <= start:
                        break
                    adjusted = dict(element)
                    adjusted["video_end"] = adjusted_end
                    try:
                        return original(adjusted)
                    except Exception:
                        continue
                raise error

        def reject_torchvision_fallback(_element: dict[str, Any]):
            raise RuntimeError(
                "ALWM strict video mode blocked qwen-vl-utils' automatic "
                "torchvision fallback after a TorchCodec decoding error"
            )

        vision_process.VIDEO_READER_BACKENDS["torchcodec"] = resilient_torchcodec
        vision_process.VIDEO_READER_BACKENDS["torchvision"] = reject_torchvision_fallback
        vision_process._alwm_strict_torchcodec_installed = True


class JsonlSceneDataset(Dataset):
    def __init__(self, manifest: str | Path, video_root: str | Path):
        self.video_root = Path(video_root)
        with Path(manifest).open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        path = Path(row["video_relpath"])
        row["video_path"] = str(path if path.is_absolute() else self.video_root / path)
        return row


def latent_prompt_parts(tokenizer: Any, question: str = DEFAULT_QUESTION) -> tuple[list[int], list[int]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{LATENT_SENTINEL}\n{question}"},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if rendered.count(LATENT_SENTINEL) != 1:
        raise RuntimeError("chat template did not preserve the latent sentinel exactly once")
    prefix, suffix = rendered.split(LATENT_SENTINEL)
    return (
        tokenizer.encode(prefix, add_special_tokens=False),
        tokenizer.encode(suffix, add_special_tokens=False),
    )


def build_caption_teacher_forcing(
    tokenizer: Any,
    latent_tokens: torch.Tensor,
    caption: str,
    question: str = DEFAULT_QUESTION,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Insert K continuous latents and label only the assistant caption."""

    if latent_tokens.ndim != 3 or latent_tokens.shape[0] != 1:
        raise ValueError("Stage-1 reference implementation expects latent_tokens [1,K,D]")
    prefix_ids, suffix_ids = latent_prompt_parts(tokenizer, question)
    eos_id = tokenizer.eos_token_id
    target_ids = tokenizer.encode(caption.strip(), add_special_tokens=False)
    if eos_id is not None:
        target_ids.append(eos_id)
    device = latent_tokens.device
    embedding = tokenizer._alwm_embedding_layer
    prefix = torch.tensor(prefix_ids, device=device, dtype=torch.long)
    suffix = torch.tensor(suffix_ids, device=device, dtype=torch.long)
    target = torch.tensor(target_ids, device=device, dtype=torch.long)
    pieces = [embedding(prefix), latent_tokens[0], embedding(suffix), embedding(target)]
    inputs_embeds = torch.cat(pieces, dim=0)[None]
    ignored = len(prefix_ids) + latent_tokens.shape[1] + len(suffix_ids)
    labels = torch.cat(
        [torch.full((ignored,), -100, device=device, dtype=torch.long), target], dim=0
    )[None]
    attention_mask = torch.ones(labels.shape, device=device, dtype=torch.long)
    return inputs_embeds, attention_mask, labels


def build_caption_teacher_forcing_batch(
    tokenizer: Any,
    latent_tokens: torch.Tensor,
    captions: list[str],
    questions: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a padded teacher-forcing batch from continuous latent tokens."""

    batch = latent_tokens.shape[0]
    if latent_tokens.ndim != 3 or batch != len(captions) or batch != len(questions):
        raise ValueError("latents, captions, and questions must have the same batch size")
    examples = [
        build_caption_teacher_forcing(
            tokenizer,
            latent_tokens[index : index + 1],
            captions[index],
            questions[index],
        )
        for index in range(batch)
    ]
    max_length = max(example[0].shape[1] for example in examples)
    padded_embeds = []
    padded_attention = []
    padded_labels = []
    for embeds, attention, labels in examples:
        padding = max_length - embeds.shape[1]
        padded_embeds.append(F.pad(embeds, (0, 0, 0, padding), value=0.0))
        padded_attention.append(F.pad(attention, (0, padding), value=0))
        padded_labels.append(F.pad(labels, (0, padding), value=-100))
    return (
        torch.cat(padded_embeds, dim=0),
        torch.cat(padded_attention, dim=0),
        torch.cat(padded_labels, dim=0),
    )


def multimodal_position_ids(batch: int, length: int, device: torch.device, offset: int = 0) -> torch.Tensor:
    positions = torch.arange(offset, offset + length, device=device, dtype=torch.long)
    return positions[None, None, :].expand(3, batch, length)


def frozen_caption_loss(
    model: Any,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Run the frozen language stack while preserving gradients to latents."""

    return frozen_caption_losses(model, inputs_embeds, attention_mask, labels).mean()


def frozen_caption_losses(
    model: Any,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return one token-normalized caption loss per sample."""

    position_ids = multimodal_position_ids(
        inputs_embeds.shape[0], inputs_embeds.shape[1], inputs_embeds.device
    )
    outputs = model.model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        return_dict=True,
    )
    logits = model.lm_head(outputs.last_hidden_state)
    token_losses = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(labels.shape[0], -1)
    valid = labels[:, 1:] != -100
    return (token_losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def _vision_messages(sample: dict[str, Any], fps: float, max_frames: int) -> list[dict[str, Any]]:
    video = {
        "type": "video",
        "video": sample["video_path"],
        "fps": fps,
        "max_frames": max_frames,
    }
    if sample.get("start_seconds") is not None and not sample.get("released_clips"):
        video["video_start"] = float(sample["start_seconds"])
    if sample.get("end_seconds") is not None and not sample.get("released_clips"):
        video["video_end"] = float(sample["end_seconds"])
    return [
        {
            "role": "user",
            "content": [
                video,
                {"type": "text", "text": "Describe this scene."},
            ],
        }
    ]


def prepare_scene_video(
    sample: dict[str, Any], fps: float = 1.0, max_frames: int = 64
) -> dict[str, Any]:
    """Decode and resize a scene interval on CPU for bounded prefetching."""

    from qwen_vl_utils import process_vision_info, vision_process

    install_strict_torchcodec_reader(vision_process)

    messages = _vision_messages(sample, fps, max_frames)
    images, videos, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        return_video_metadata=False,
        image_patch_size=16,
    )
    if isinstance(video_kwargs.get("fps"), list) and len(video_kwargs["fps"]) == 1:
        video_kwargs["fps"] = video_kwargs["fps"][0]
    return {
        "messages": messages,
        "images": images,
        "videos": videos,
        "video_kwargs": video_kwargs,
    }


def prepare_scene_model_inputs(
    sample: dict[str, Any],
    processor: Any,
    fps: float = 1.0,
    max_frames: int = 64,
) -> dict[str, torch.Tensor]:
    """Decode, resize, normalize, and patchify a scene entirely on CPU."""

    prepared = prepare_scene_video(sample, fps, max_frames)
    inputs = processor.video_processor(
        videos=prepared["videos"],
        return_tensors="pt",
        do_resize=False,
        do_sample_frames=False,
    )
    return {
        "pixel_values_videos": inputs["pixel_values_videos"],
        "video_grid_thw": inputs["video_grid_thw"],
    }


def encode_scene_tokens(
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    device: torch.device,
    fps: float = 1.0,
    max_frames: int = 64,
    return_metadata: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
]:
    """Encode one requested interval and return merged LLM-space tokens."""

    result = encode_scene_tokens_batch(
        model,
        processor,
        [sample],
        device,
        fps,
        max_frames,
        return_metadata=return_metadata,
    )
    if not return_metadata:
        return result
    features, times, mask, metadata = result
    return features, times, mask, metadata[0]


def encode_scene_tokens_batch(
    model: Any,
    processor: Any,
    samples: list[dict[str, Any]],
    device: torch.device,
    fps: float = 1.0,
    max_frames: int = 64,
    return_metadata: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]
]:
    """Batch variable-length videos through the frozen visual encoder once."""

    if not samples:
        raise ValueError("samples must not be empty")
    prepared_batch = []
    for sample in samples:
        prepared = sample.pop("_alwm_prepared_model_inputs", None)
        if prepared is None:
            prepared = prepare_scene_model_inputs(sample, processor, fps, max_frames)
        prepared_batch.append(prepared)
    pixel_values = torch.cat(
        [prepared["pixel_values_videos"] for prepared in prepared_batch], dim=0
    ).to(device=device, dtype=model.dtype)
    grid = torch.cat(
        [prepared["video_grid_thw"] for prepared in prepared_batch], dim=0
    ).to(device)

    def unpack_feature_rows(encoded: Any) -> list[torch.Tensor]:
        values = encoded.pooler_output if hasattr(encoded, "pooler_output") else encoded[0]
        if torch.is_tensor(values):
            return [values[0] if values.ndim == 3 else values]
        return list(values)

    with torch.no_grad():
        feature_rows = unpack_feature_rows(model.get_video_features(pixel_values, grid))
    if len(feature_rows) != len(samples):
        raise RuntimeError(
            f"visual encoder returned {len(feature_rows)} feature groups for {len(samples)} videos"
        )

    feature_rows = [features.detach() for features in feature_rows]
    maximum_tokens = max(features.shape[0] for features in feature_rows)
    hidden_size = feature_rows[0].shape[-1]
    features = feature_rows[0].new_zeros((len(samples), maximum_tokens, hidden_size))
    times = feature_rows[0].new_zeros((len(samples), maximum_tokens))
    mask = torch.zeros((len(samples), maximum_tokens), device=device, dtype=torch.bool)
    metadata_rows = []
    merge = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
    for index, row_features in enumerate(feature_rows):
        count = row_features.shape[0]
        features[index, :count] = row_features
        mask[index, :count] = True
        t, h, w = [int(value) for value in grid[index].tolist()]
        tokens_per_step = max(1, (h * w) // (merge * merge))
        row_times = torch.linspace(
            0.0, 1.0, t, device=device, dtype=row_features.dtype
        ).repeat_interleave(tokens_per_step)
        if row_times.numel() != count:
            row_times = torch.linspace(0.0, 1.0, count, device=device, dtype=row_features.dtype)
        times[index, :count] = row_times
        metadata_rows.append(
            {
                "native_visual_tokens": int(count),
                "video_grid_thw": [t, h, w],
                "temporal_steps_after_patchify": t,
                "tokens_per_temporal_step": tokens_per_step,
                "requested_fps": float(fps),
                "max_frames": int(max_frames),
            }
        )
    result = (features, times, mask)
    if not return_metadata:
        return result
    return (*result, metadata_rows)


def _stop_token_ids(tokenizer: Any) -> set[int]:
    values = tokenizer.eos_token_id
    stop_ids = set(values if isinstance(values, (list, tuple)) else [values])
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id != tokenizer.unk_token_id:
            stop_ids.add(token_id)
    return {int(value) for value in stop_ids if value is not None}


def attach_embedding_layer(tokenizer: Any, model: Any) -> None:
    """Avoid threading the frozen embedding module through every prompt helper."""

    tokenizer._alwm_embedding_layer = model.get_input_embeddings()
