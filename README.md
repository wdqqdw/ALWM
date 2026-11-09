# ALWM

**Accessible Latent World Modeling for Predictive Video Reasoning**

Inference and supervised training for ALWM, based on Qwen3-VL-8B-Instruct.

- [Model](https://huggingface.co/wudq/ALWM)
- [Data](https://huggingface.co/datasets/wudq/ALWM_data)

## Setup

Use Linux, Python 3.12, a CUDA GPU, and FFmpeg (`ffmpeg` and `ffprobe` on `PATH`).
For a CUDA 13.0-compatible driver, create an isolated environment and install:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.13.0 torchvision==0.28.0 torchcodec==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

These [PyTorch CUDA wheels](https://download.pytorch.org/whl/cu130/) were tested
with Transformers 4.57.6 and qwen-vl-utils 0.0.14. If you use another CUDA build,
keep PyTorch, torchvision, and TorchCodec ABI-compatible.
Inference loads the FP32 release in BF16 without modifying its files.
Training defaults to two GPUs with one full model replica per GPU; its memory
requirements are higher than single-GPU inference.

## Download and prepare data

Run commands from the repository root:

```bash
hf download wudq/ALWM --local-dir models/ALWM
hf download wudq/ALWM_data --repo-type dataset --local-dir data/ALWM_data

# Needed for SFT, not for inference with the released model.
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
```

Keep the downloaded directory structure unchanged. Video paths in each metadata
file are relative to `data/ALWM_data/`, not to the metadata file's directory.

```text
ALWM/
├── README.md
├── inference/
├── training/
├── models/
│   ├── ALWM/
│   └── Qwen3-VL-8B-Instruct/
└── data/
    └── ALWM_data/
        ├── D1/
        │   ├── D1_metadata.json
        │   └── videos/000/*.mp4
        ├── D2/
        │   ├── D2_metadata.json
        │   └── videos/000/*.mp4
        └── D3/
            ├── D3_metadata.json
            └── videos/000/*.mp4
```

`000/` is the first numbered video directory; keep all numbered directories.
The full dataset is approximately 590 GB. Allow additional disk space for models
and training checkpoints.

| Split | Samples | Contents |
| --- | ---: | --- |
| D1 | 42,673 | Video and caption for compressor training |
| D2 | 15,290 | Separate observation/future clips and tagged text targets |
| D3 | 2,885 | Observation clip, question/options, and tagged answer target |

Before training, validate the metadata and generate JSONL manifests:

```bash
bash training/prepare_data.sh
```

This checks sample counts, video paths, durations, and answer-to-option mappings,
then writes `data/manifests/{D1,D2,D3}.jsonl` and `audit.json`. It preserves IDs,
text, and labels. The videos are already trimmed: do not crop them again.
`<latent></latent>` is a marker, not a stored latent vector; training computes
continuous states on the fly. D3 needs no future video. Data preparation can
take several minutes; it does not copy the videos.

## Inference

Run the first D3 example:

```bash
bash inference/run.sh
```

Or select an example / provide your own observed video:

```bash
bash inference/run.sh --sample-id YOUR_D3_SAMPLE_ID --output outputs/example.json

bash inference/run.sh \
  --video /absolute/path/to/observed.mp4 \
  --question "What is most likely to happen next?" \
  --output outputs/custom.json
```

The script uses only the observed video and question/options, never the target
answer. It generates reasoning, rolls out eight continuous latent states with
the same KV cache, and generates the prediction (plus an answer for MCQs).
The default result is `outputs/inference/result.json`; choose a new output path
for another run. An incomplete sequence returns a nonzero exit code.

## Reproduce SFT

Start from **Qwen3-VL-8B-Instruct**, not from the released ALWM checkpoint.
Run all four stages in order:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash training/run_all.sh
```

Or run the same stages separately:

```bash
bash training/run_compressor.sh
bash training/run_p1.sh
bash training/run_p2.sh
bash training/run_p3.sh
```

| Stage | Data | Training | Output |
| --- | --- | --- | --- |
| Compressor | D1 | Train an 8-slot C1 compressor; freeze Qwen | `outputs/compressor/` |
| P1 | D2 | Text supervision and future-latent alignment, through step 2,500 | `outputs/p1/checkpoints/step-002500/` |
| P2 | D2 | Add autoregressive latent rollout and reset-context caption loss, through step 8,500 | `outputs/p2/checkpoints/step-008500/` |
| P3 | D3 + 20% D2 rehearsal | Continuous-context SFT, through step 10,304 | `outputs/p3/checkpoints/step-010304/` |

P1–P3 train the language model and LM head while freezing the visual encoder and
compressor. Each stage loads its predecessor automatically, including optimizer
and RNG state for P2/P3. The launchers contain the learning rates, seeds, frame
budgets, and schedules; `DRY_RUN=1 bash training/run_all.sh` prints the commands.
Logs and checkpoints stay under `outputs/`. Existing stage output directories
are not overwritten. Use a fresh `OUTPUT_ROOT` for a new run.

Run inference with your resulting SFT checkpoint:

```bash
MODEL_DIR=models/Qwen3-VL-8B-Instruct bash inference/run.sh \
  --sft-checkpoint outputs/p3/checkpoints/step-010304 \
  --output outputs/sft-inference.json
```

Paths can be changed with `MODEL_DIR` (inference), `BASE_MODEL_DIR` (training),
`DATA_ROOT`, `MANIFEST_DIR`, and `OUTPUT_ROOT`. `PYTHON_BIN` selects the Python
interpreter. The training recipe uses `NPROC_PER_NODE=2`; changing the world size
also changes the effective batch size.

This reproduces the pre-RL training pipeline using the released clips, not a
bit-identical copy of the released model. The release does not contain a separate
held-out split; training examples are not reported as held-out evaluation.

Offline data-contract and four-stage gradient tests:

```bash
python training/test_pipeline.py
```

Validation: both released-model and SFT-checkpoint inference completed on a 32 GB GPU with eight finite
4096-dimensional latent states (17.8 GiB peak allocated memory). A real compressor
optimizer step and all six offline tests passed. All 60,848 metadata records and
their video paths were checked. These are smoke tests, not a new full SFT run or
an accuracy evaluation.

## RL

For RL, please use the RL data in [V1-33K](https://huggingface.co/datasets/haonan3/V1-33K).

Start from the SFT checkpoint and follow the standard GRPO workflow: sample
multiple responses for each observed-video/question pair, score the rollouts,
compute group-relative advantages, and update the policy with a clipped objective
and reference-model KL regularization.

Apply the response/loss mask only to sampled discrete response tokens. Set it to
zero for the eight continuous latent slots and padding; prompt tokens are also
excluded from the loss. Use the same mask for policy-gradient, KL, and entropy
terms. Keep the latent states in the causal attention context: they are not
vocabulary actions, and the loss mask must not remove them from attention.
