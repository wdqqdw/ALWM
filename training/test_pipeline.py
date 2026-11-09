"""Offline contract and gradient tests; uses a tiny real Qwen3-VL text stack."""
import json
import random
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from prepare_data import convert_row, parse_text, safe_video
from alwm.compressors import build_compressor
from alwm.stage1 import attach_embedding_layer, build_caption_teacher_forcing, frozen_caption_loss
from alwm.stage2 import (
    CurriculumDataset, FutureLatentTrainingModule, build_future_prefix,
    rollout_future_latents, continuous_sft_objective, wrapped_training_text,
)
from train_curriculum import save_checkpoint, load_checkpoint


class TinyTokenizer:
    eos_token_id = 2
    def encode(self, text, **kwargs):
        return [3 + ord(char) % 120 for char in text]
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages) + "\nassistant:"


class TinyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        config = Qwen3VLTextConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]})
        config._attn_implementation = "sdpa"
        self.model = nn.Module()
        self.model.language_model = Qwen3VLTextModel(config)
        self.lm_head = nn.Linear(32, 128, bias=False)
    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens


def target(gold=None):
    text = "<observation>Seen.</observation>\n<reasoning>Next.</reasoning>\n<latent></latent>\n<prediction>Walk.</prediction>"
    return text + (f"<answer>{gold}</answer>" if gold else "")


class DataTests(unittest.TestCase):
    def test_d3_labels_and_no_future(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "observed.mp4").write_bytes(b"fixture")
            item = {"video": "observed.mp4", "query": "Next?\n\nOptions:\nA. Walk.\nB. Run.\nC. Sit.\nD. Wait.", "answer": target("A")}
            row = convert_row(root, "D3", "sample", item, probe=False)
            self.assertIsNone(row["future"])
            self.assertEqual(row["answer"], "A")
            self.assertEqual(row["options"]["A"], "Walk.")
            self.assertTrue(wrapped_training_text(row).endswith("<prediction>Walk.</prediction><answer>A</answer>"))
            manifest = root / "d3.jsonl"
            manifest.write_text(json.dumps(row) + "\n")
            loaded = CurriculumDataset(manifest, root)[0]
            self.assertNotIn("future_scene", loaded)
            self.assertIsNone(loaded["context_scene"]["end_seconds"])
            item["answer"] = target("B")
            with self.assertRaises(ValueError): parse_text("D3", "sample", item)

    def test_d2_distinct_pretrimmed_clips(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("observed.mp4", "future.mp4"): (root / name).write_bytes(b"fixture")
            row = convert_row(root, "D2", "sample", {"video": {"observation": "observed.mp4", "future": "future.mp4"}, "query": "Next?", "answer": target()}, probe=False)
            manifest = root / "d2.jsonl"
            manifest.write_text(json.dumps(row) + "\n")
            loaded = CurriculumDataset(manifest, root)[0]
            self.assertNotEqual(loaded["context_scene"]["video_path"], loaded["future_scene"]["video_path"])
            self.assertIsNone(loaded["future_scene"]["start_seconds"])

    def test_reject_escaping_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            for path in ("../outside.mp4", "/outside.mp4"):
                with self.assertRaises(ValueError): safe_video(Path(temp), path)


class GradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)
        self.model, self.tokenizer = TinyQwen(), TinyTokenizer()
        attach_embedding_layer(self.tokenizer, self.model)
        self.observed = torch.randn(1, 4, 32)
        self.teacher = torch.randn(1, 8, 32)
        self.row = {"question": "Next?", "video_understanding": "Seen.", "future_reasoning": "Next.", "future_event_summary": "Walk."}

    def test_compressor_backward_through_frozen_language(self):
        self.model.requires_grad_(False)
        compressor = build_compressor("c1", 32, 8, 16)
        latents = compressor(self.observed, torch.linspace(0, 1, 4)[None], torch.ones(1, 4, dtype=torch.bool))
        self.assertEqual(tuple(latents.shape), (1, 8, 32))
        embeds, attention, labels = build_caption_teacher_forcing(self.tokenizer, latents, "Walk.")
        loss = frozen_caption_loss(self.model, embeds, attention, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in compressor.parameters() if p.grad is not None), 0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_p1_p2_p3_optimizer_steps(self):
        module = FutureLatentTrainingModule(self.model, self.tokenizer)
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4)
        for phase in ("a", "b", "c"):
            optimizer.zero_grad(set_to_none=True)
            metrics = module(phase, self.observed, None if phase == "c" else self.teacher, self.row)
            loss = metrics["loss"]
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
            optimizer.step()
            if phase == "c":
                self.assertNotIn("alignment_loss", metrics)
                self.assertEqual(float(metrics["kv_reset"]), 0)
        with self.assertRaises(ValueError): module("a", self.observed, None, self.row)
        # The phase hand-off must restore weights, optimizer moments, and RNG.
        with tempfile.TemporaryDirectory() as temp:
            saved = {name: value.detach().clone() for name, value in module.state_dict().items()}
            path = save_checkpoint(Path(temp), module, optimizer,
                {"global_step": 3, "micro_step": 3}, 0, 1)
            expected_random = (random.random(), torch.rand(4))
            with torch.no_grad():
                for parameter in module.parameters(): parameter.add_(1)
            state = load_checkpoint(path, module, optimizer, torch.device("cpu"))
            self.assertEqual(state, {"global_step": 3, "micro_step": 3})
            self.assertEqual(random.random(), expected_random[0])
            self.assertTrue(torch.equal(torch.rand(4), expected_random[1]))
            for name, value in module.state_dict().items():
                self.assertTrue(torch.equal(value, saved[name]), name)
            self.assertTrue(optimizer.state)

    def test_p3_gradient_reaches_all_eight_states(self):
        prefix = build_future_prefix(self.model, self.tokenizer, self.observed, self.row)
        latents = rollout_future_latents(self.model, prefix, 8, "recompute")
        latents.retain_grad()
        continuous_sft_objective(self.model, self.tokenizer, prefix, latents, self.row)["post_text_ce"].backward()
        self.assertEqual(tuple(latents.shape), (1, 8, 32))
        self.assertTrue((latents.grad.abs().sum(dim=-1) > 0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
