"""P1/P2/P3 future-latent SFT objectives.

Observed frames are encoded by Qwen3-VL's native visual encoder and their
uncompressed visual tokens enter the language model directly.  C1 is used only
on the actual-future teacher branch.  P2 uses a reset caption objective; phase C
is a single continuous observation/reasoning/latent/prediction sequence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .stage1 import (
    DEFAULT_QUESTION,
    SYSTEM_PROMPT,
    build_caption_teacher_forcing_batch,
    frozen_caption_losses,
    multimodal_position_ids,
)


OBSERVED_VIDEO_SENTINEL = "<<ALWM_OBSERVED_VIDEO_TOKENS>>"
FUTURE_LATENT_OPEN = "<latent>"
FUTURE_LATENT_CLOSE = "</latent>"
PREDICTION_OPEN = "<prediction>"
PREDICTION_CLOSE = "</prediction>"
LATENT_PLACEHOLDER = "<LTT>"
DEFAULT_FUTURE_QUESTION = "What is most likely to happen after the observed video context?"
CHOICE_LETTERS = ("A", "B", "C", "D")


@dataclass
class FuturePrefix:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    text_labels: torch.Tensor
    assistant_text: str


class CurriculumDataset(Dataset):
    def __init__(self, manifest: str | Path, video_root: str | Path):
        self.video_root = Path(video_root)
        with Path(manifest).open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        path = Path(row["video_relpath"])
        video_path = str(path if path.is_absolute() else self.video_root / path)
        if not row.get("released_clips"):
            raise ValueError("use manifests from training/prepare_data.sh")
        row["context_scene"] = {"video_path": video_path, "start_seconds": None, "end_seconds": None}
        if row.get("future") is not None:
            row["future_scene"] = {
                "video_path": str(self.video_root / row["future"]["video_relpath"]),
                "start_seconds": None, "end_seconds": None,
            }
        return row


def _language_model(model: Any) -> nn.Module:
    return model.model.language_model


def _embedding(model: Any) -> nn.Module:
    return model.get_input_embeddings()


def _token_ids(tokenizer: Any, text: str, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        tokenizer.encode(text, add_special_tokens=False), device=device, dtype=torch.long
    )


def assistant_reasoning_text(row: dict[str, Any]) -> str:
    return (
        f"<observation>{row['video_understanding'].strip()}</observation>\n"
        f"<reasoning>{row['future_reasoning'].strip()}</reasoning>\n"
        f"{FUTURE_LATENT_OPEN}"
    )


def multiple_choice_fields(row: dict[str, Any]) -> tuple[dict[str, str], str] | None:
    """Validate D3 multiple-choice supervision; D2 has text-only targets."""

    options = row.get("options")
    answer = row.get("answer")
    if options is None and answer is None:
        return None
    if not isinstance(options, dict) or set(options) != set(CHOICE_LETTERS):
        raise ValueError("options must cover exactly A/B/C/D when answer is present")
    normalized_options: dict[str, str] = {}
    for letter in CHOICE_LETTERS:
        value = options[letter]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"option {letter} must be non-empty text")
        normalized_options[letter] = value.strip()
    if answer not in CHOICE_LETTERS:
        raise ValueError("answer must be one of A/B/C/D")
    prediction = str(row["future_event_summary"]).strip()
    if prediction != normalized_options[answer]:
        raise ValueError("future_event_summary must equal the answered option")
    return normalized_options, str(answer)


def future_question_text(row: dict[str, Any]) -> str:
    """Render the observed-video question, including structured MC options."""

    question = str(row.get("question") or DEFAULT_FUTURE_QUESTION).strip()
    multiple_choice = multiple_choice_fields(row)
    if multiple_choice is None:
        return question
    options, _ = multiple_choice
    option_text = "\n".join(f"{letter}. {options[letter]}" for letter in CHOICE_LETTERS)
    return f"{question}\n\nOptions:\n{option_text}"


def prediction_and_answer_text(row: dict[str, Any]) -> str:
    """Serialize post-latent supervision with answer as a prediction sibling."""

    prediction = str(row["future_event_summary"]).strip()
    text = f"{PREDICTION_OPEN}{prediction}{PREDICTION_CLOSE}"
    multiple_choice = multiple_choice_fields(row)
    if multiple_choice is not None:
        _, answer = multiple_choice
        text += f"<answer>{answer}</answer>"
    return text


def wrapped_training_text(row: dict[str, Any], latent_tokens: int = 8) -> str:
    """Human-readable serialization of the mixed text/continuous sequence."""

    if latent_tokens < 1:
        raise ValueError("latent_tokens must be positive")
    return (
        f"<observation>{row['video_understanding'].strip()}</observation>"
        f"<reasoning>{row['future_reasoning'].strip()}</reasoning>"
        f"<latent>{LATENT_PLACEHOLDER * latent_tokens}</latent>"
        f"{prediction_and_answer_text(row)}"
    )


def build_future_prefix(
    model: Any,
    tokenizer: Any,
    observed_video_tokens: torch.Tensor,
    row: dict[str, Any],
) -> FuturePrefix:
    """Build raw observed-video tokens + teacher-forced text to latent start.

    Caption labels cover the complete assistant observation/reasoning text,
    including the explicit future-latent start marker.  User/system tokens and
    uncompressed visual tokens are ignored by text CE.
    """

    if observed_video_tokens.ndim != 3 or observed_video_tokens.shape[0] != 1:
        raise ValueError("observed_video_tokens must be [1,N,D]")
    question = future_question_text(row)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{OBSERVED_VIDEO_SENTINEL}\n{question}"},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if rendered.count(OBSERVED_VIDEO_SENTINEL) != 1:
        raise RuntimeError("chat template did not preserve the observed-video sentinel once")
    prefix_text, suffix_text = rendered.split(OBSERVED_VIDEO_SENTINEL)
    assistant_text = assistant_reasoning_text(row)
    device = observed_video_tokens.device
    prefix_ids = _token_ids(tokenizer, prefix_text, device)
    suffix_ids = _token_ids(tokenizer, suffix_text, device)
    assistant_ids = _token_ids(tokenizer, assistant_text, device)
    embedding = _embedding(model)
    inputs = torch.cat(
        [
            embedding(prefix_ids),
            observed_video_tokens[0],
            embedding(suffix_ids),
            embedding(assistant_ids),
        ],
        dim=0,
    )[None]
    ignored = prefix_ids.numel() + observed_video_tokens.shape[1] + suffix_ids.numel()
    labels = torch.cat(
        [
            torch.full((ignored,), -100, device=device, dtype=torch.long),
            assistant_ids,
        ]
    )[None]
    attention = torch.ones(labels.shape, device=device, dtype=torch.long)
    return FuturePrefix(inputs, attention, labels, assistant_text)


def token_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    losses = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(labels.shape[0], -1)
    valid = labels[:, 1:] != -100
    per_example = (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    return per_example.mean(), valid.sum()


def slotwise_cosine(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted/target latent shapes differ: {tuple(predicted.shape)} vs {tuple(target.shape)}"
        )
    return F.cosine_similarity(predicted.float(), target.float(), dim=-1).mean()


def teacher_forced_future_latents(
    model: Any,
    prefix: FuturePrefix,
    target_latents: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Teacher-force K target slots and predict each slot with shift-right."""

    if target_latents.ndim != 3 or target_latents.shape[0] != 1:
        raise ValueError("target_latents must be [1,K,D]")
    k = target_latents.shape[1]
    inputs = torch.cat([prefix.inputs_embeds, target_latents[:, :-1]], dim=1)
    extra = torch.ones((1, k - 1), device=inputs.device, dtype=prefix.attention_mask.dtype)
    attention = torch.cat([prefix.attention_mask, extra], dim=1)
    latent_labels = torch.full((1, k - 1), -100, device=inputs.device, dtype=torch.long)
    labels = torch.cat([prefix.text_labels, latent_labels], dim=1)
    outputs = _language_model(model)(
        inputs_embeds=inputs,
        attention_mask=attention,
        position_ids=multimodal_position_ids(1, inputs.shape[1], inputs.device),
        use_cache=False,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        return_dict=True,
    )
    logits = model.lm_head(outputs.last_hidden_state)
    text_ce, text_tokens = token_cross_entropy(logits, labels)
    prefix_length = prefix.inputs_embeds.shape[1]
    indices = torch.arange(
        prefix_length - 1,
        prefix_length + k - 1,
        device=inputs.device,
        dtype=torch.long,
    )
    predicted = outputs.last_hidden_state.index_select(1, indices)
    cosine = slotwise_cosine(predicted, target_latents)
    return {
        "text_ce": text_ce,
        "text_tokens": text_tokens,
        "predicted_latents": predicted,
        "latent_cosine": cosine,
        "alignment_loss": 1.0 - cosine,
    }


def _rollout_future_latents_recompute(
    model: Any,
    prefix: FuturePrefix,
    latent_tokens: int,
) -> torch.Tensor:
    """Differentiably roll out continuous hidden states without token decode.

    Recomputing the full prefix at each of the eight steps avoids a mutable KV
    cache and stays compatible with gradient checkpointing.  These passes are
    context-conditioned by design: they produce the future latent.  The later
    B caption objective deliberately starts a separate, reset sequence.
    """

    if latent_tokens < 1:
        raise ValueError("latent_tokens must be positive")
    sequence = prefix.inputs_embeds
    generated: list[torch.Tensor] = []
    for index in range(latent_tokens):
        attention = torch.ones(sequence.shape[:2], device=sequence.device, dtype=torch.long)
        outputs = _language_model(model)(
            inputs_embeds=sequence,
            attention_mask=attention,
            position_ids=multimodal_position_ids(1, sequence.shape[1], sequence.device),
            use_cache=False,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            return_dict=True,
        )
        next_latent = outputs.last_hidden_state[:, -1:]
        generated.append(next_latent)
        if index + 1 < latent_tokens:
            sequence = torch.cat([sequence, next_latent], dim=1)
    return torch.cat(generated, dim=1)


def rollout_future_latents(
    model: Any,
    prefix: FuturePrefix,
    latent_tokens: int,
    backend: str = "recompute",
) -> torch.Tensor:
    """Roll out continuous states while retaining gradients through all slots."""
    if backend != "recompute":
        raise ValueError("SFT uses the recompute backend")
    return _rollout_future_latents_recompute(model, prefix, latent_tokens)


def continuous_sft_objective(
    model: Any,
    tokenizer: Any,
    prefix: FuturePrefix,
    latent_tokens: torch.Tensor,
    row: dict[str, Any],
) -> dict[str, torch.Tensor | bool]:
    """Supervise pre/post text in one continuous sequence with no KV reset."""

    if latent_tokens.ndim != 3 or latent_tokens.shape[0] != 1:
        raise ValueError("latent_tokens must be [1,K,D]")
    device = latent_tokens.device
    post_text = f"{FUTURE_LATENT_CLOSE}\n{prediction_and_answer_text(row)}"
    post_ids = tokenizer.encode(post_text, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        post_ids.append(int(eos_id))
    post_tensor = torch.tensor(post_ids, device=device, dtype=torch.long)
    inputs = torch.cat(
        [prefix.inputs_embeds, latent_tokens, _embedding(model)(post_tensor)[None]],
        dim=1,
    )
    latent_and_post = latent_tokens.shape[1] + post_tensor.numel()
    pre_labels = torch.cat(
        [
            prefix.text_labels,
            torch.full(
                (1, latent_and_post), -100, device=device, dtype=torch.long
            ),
        ],
        dim=1,
    )
    post_labels = torch.cat(
        [
            torch.full(
                (1, prefix.inputs_embeds.shape[1] + latent_tokens.shape[1]),
                -100,
                device=device,
                dtype=torch.long,
            ),
            post_tensor[None],
        ],
        dim=1,
    )
    attention = torch.ones(inputs.shape[:2], device=device, dtype=torch.long)
    outputs = _language_model(model)(
        inputs_embeds=inputs,
        attention_mask=attention,
        position_ids=multimodal_position_ids(1, inputs.shape[1], device),
        use_cache=False,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        return_dict=True,
    )
    logits = model.lm_head(outputs.last_hidden_state)
    pre_ce, pre_tokens = token_cross_entropy(logits, pre_labels)
    post_ce, post_tokens = token_cross_entropy(logits, post_labels)
    return {
        "pre_text_ce": pre_ce,
        "pre_text_tokens": pre_tokens,
        "post_text_ce": post_ce,
        "post_text_tokens": post_tokens,
        "continuous_context": True,
        "reset_context_kv": False,
    }


def reset_caption_objective(
    model: Any,
    tokenizer: Any,
    predicted_latents: torch.Tensor,
    captions: list[str],
) -> dict[str, torch.Tensor | bool | str]:
    """Stage-1-identical latent-only caption objective with a fresh context.

    No earlier prefix, raw video, reasoning hidden state, or past key/value is
    accepted by this function.  The exact Stage-1 question is imported rather
    than copied, and Stage-1's forward uses ``use_cache=False``.
    """

    questions = [DEFAULT_QUESTION] * len(captions)
    inputs, attention, labels = build_caption_teacher_forcing_batch(
        tokenizer, predicted_latents, captions, questions
    )
    losses = frozen_caption_losses(model, inputs, attention, labels)
    return {
        "caption_loss": losses.mean(),
        "caption_losses": losses,
        "caption_tokens": (labels[:, 1:] != -100).sum(),
        "reset_context_kv": True,
        "question": DEFAULT_QUESTION,
    }


class FutureLatentTrainingModule(nn.Module):
    """DDP-safe wrapper containing every trainable Qwen language parameter."""

    def __init__(
        self,
        qwen: nn.Module,
        tokenizer: Any,
        alignment_weight: float = 1.0,
        caption_weight: float = 1.0,
        post_text_weight: float = 1.0,
        rollout_backend: str = "recompute",
    ):
        super().__init__()
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.alignment_weight = float(alignment_weight)
        self.caption_weight = float(caption_weight)
        self.post_text_weight = float(post_text_weight)
        self.rollout_backend = rollout_backend


    def forward(
        self,
        phase: str,
        observed_video_tokens: torch.Tensor,
        target_latents: torch.Tensor | None,
        row: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        normalized = phase.lower()
        if normalized not in {"a", "b", "c"}:
            raise ValueError("phase must be A, B, or C")
        if normalized in {"a", "b"} and target_latents is None:
            raise ValueError("P1/P2 require actual-future teacher latents")
        prefix = build_future_prefix(
            self.qwen, self.tokenizer, observed_video_tokens, row
        )
        metrics: dict[str, torch.Tensor] = {}
        if normalized == "a":
            teacher = teacher_forced_future_latents(
                self.qwen, prefix, target_latents
            )
            metrics["pre_text_ce"] = teacher["text_ce"]
            metrics["teacher_latent_cosine"] = teacher["latent_cosine"]
            metrics["alignment_loss"] = teacher["alignment_loss"]
            metrics["weighted_alignment_loss"] = (
                self.alignment_weight * teacher["alignment_loss"]
            )
            metrics["loss"] = teacher["text_ce"] + metrics["weighted_alignment_loss"]
            return metrics

        if normalized == "b":
            teacher = teacher_forced_future_latents(
                self.qwen, prefix, target_latents
            )
            metrics["pre_text_ce"] = teacher["text_ce"]
            metrics["teacher_latent_cosine"] = teacher["latent_cosine"]
            metrics["alignment_loss"] = teacher["alignment_loss"]
            metrics["weighted_alignment_loss"] = (
                self.alignment_weight * teacher["alignment_loss"]
            )
            rolled = rollout_future_latents(
                self.qwen, prefix, target_latents.shape[1], self.rollout_backend
            )
            metrics["rollout_latent_cosine"] = slotwise_cosine(
                rolled, target_latents
            ).detach()
            caption = reset_caption_objective(
                self.qwen,
                self.tokenizer,
                rolled,
                [row["future_event_summary"]],
            )
            metrics["reset_caption_ce"] = caption["caption_loss"]
            metrics["weighted_reset_caption_loss"] = (
                self.caption_weight * caption["caption_loss"]
            )
            metrics["p2_autoregressive_latent"] = target_latents.new_ones(
                (), dtype=torch.float32
            )
            metrics["p2_reset_prediction_ce"] = target_latents.new_ones(
                (), dtype=torch.float32
            )
            metrics["loss"] = (
                teacher["text_ce"]
                + metrics["weighted_alignment_loss"]
                + metrics["weighted_reset_caption_loss"]
            )
            return metrics

        post_latents = rollout_future_latents(self.qwen, prefix, 8, self.rollout_backend)
        post = continuous_sft_objective(
            self.qwen, self.tokenizer, prefix, post_latents, row
        )
        metrics["pre_text_ce"] = post["pre_text_ce"]
        metrics["post_text_ce"] = post["post_text_ce"]
        metrics["continuous_context"] = observed_video_tokens.new_ones(
            (), dtype=torch.float32
        )
        metrics["kv_reset"] = observed_video_tokens.new_zeros((), dtype=torch.float32)
        metrics["weighted_post_text_loss"] = (
            self.post_text_weight * post["post_text_ce"]
        )
        metrics["loss"] = post["pre_text_ce"] + metrics["weighted_post_text_loss"]
        return metrics
