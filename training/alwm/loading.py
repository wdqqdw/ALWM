"""Read local Qwen processor exports without changing their vocabulary."""
import json
from pathlib import Path


def load_processor(directory):
    from transformers import AutoProcessor

    config = json.loads((Path(directory) / "tokenizer_config.json").read_text())
    extra = config.get("extra_special_tokens")
    overrides = {}
    if isinstance(extra, list):
        # Transformers 5 serializes this list under extra_special_tokens;
        # Transformers 4 expects that field to be a named-token dictionary.
        special = list(dict.fromkeys(config.get("additional_special_tokens", []) + extra))
        overrides = {"extra_special_tokens": {}, "additional_special_tokens": special}
    processor = AutoProcessor.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False, **overrides
    )
    tokenizer = processor.tokenizer
    for token_id, details in config.get("added_tokens_decoder", {}).items():
        if tokenizer.convert_tokens_to_ids(details["content"]) != int(token_id):
            raise ValueError("processor compatibility changed a checkpoint token ID")
    if isinstance(extra, list):
        if not set(extra).issubset(set(tokenizer.all_special_tokens)):
            raise ValueError("checkpoint special tokens were not preserved")
    return processor


def load_inference_model(directory, dtype):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    raw = json.loads((Path(directory) / "config.json").read_text())
    text = raw["text_config"]
    if text.get("rope_parameters") and not text.get("rope_scaling"):
        # The v5 field combines v4's rope_scaling and rope_theta. This is a
        # serialization migration, not a change to positional encoding.
        rope = dict(text.pop("rope_parameters"))
        text["rope_theta"] = rope.pop("rope_theta")
        text["rope_scaling"] = rope
    config = Qwen3VLConfig.from_dict(raw)
    model, info = Qwen3VLForConditionalGeneration.from_pretrained(
        directory, config=config, dtype=dtype, attn_implementation="sdpa",
        local_files_only=True, trust_remote_code=False, output_loading_info=True,
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"checkpoint did not load exactly: {info}")
    return model
