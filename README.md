<img src="./mimic-video.png" width="450px"></img>

## Mimic-Video: LIBERO Training & Evaluation

A robot manipulation policy learning pipeline built on [Mimic-Video](https://mimic-video.github.io/), combining a Cosmos 2.0 video backbone with a flow-matching diffusion action head.

## Architecture

```
Observation (front + wrist images)
        │
        ▼
┌─────────────────────────┐
│  Cosmos 2.0 Backbone    │  nvidia/Cosmos-Predict2-2B-Video2World
│  (6.9B params, frozen)  │  Extract video hidden states from layer 19
└──────────┬──────────────┘
           │ video features (cross-attention context)
           ▼
┌─────────────────────────┐
│  MimicVideo DiT Head    │  8-layer DiT, dim=512, 8 heads
│  (73M trainable params) │
│                         │
│  Query: [state_token(1) + action_tokens(16)]
│  Context: video_hiddens (cross-attention)
│  Conditioning: timestep → Fourier → AdaptiveRMSNorm
└──────────┬──────────────┘
           │
           ▼
    Action trajectory (16 steps × 7D)
```

- **Action head**: joint_state (1 token) + noised actions (16 tokens) cross-attend to video features
- **Flow matching**: Trained with velocity prediction, sampled via DDPM-style denoising
- **Normalization**: Actions/states are mean-std normalized, padded to max_action_dim=32 / max_state_dim=64 (LIBERO uses 7 real dims)

## Setup

```bash
# Create conda environment
conda create -n mimic_video python=3.10
conda activate mimic_video

# Install packages
pip install -e '.[test]'
pip install -r requirements.txt

# Or all at once
bash install.sh
```

## Training

### Single GPU

```bash
python scripts/train.py \
    --config configs/libero.yaml \
    --dataset_path /path/to/libero_gr00t_delta \
    --output_dir ./checkpoints/libero \
    --max_steps 60000 \
    --per_device_train_batch_size 16 \
    --freeze_video_backbone \
    --report_to wandb
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/libero.yaml \
    --dataset_path /path/to/libero_gr00t_delta \
    --output_dir ./checkpoints/libero \
    --max_steps 60000 \
    --per_device_train_batch_size 16 \
    --freeze_video_backbone \
    --report_to wandb
```

In multi-GPU mode, rank 0 downloads and caches the Cosmos model first, then a barrier ensures other ranks load from cache, preventing HuggingFace Hub race conditions.

### SLURM (sbatch)

```bash
# Default (60k steps, batch=16, 2 GPUs)
sbatch run_scripts/train_libero_cosmos2.sh

# Override via environment variables
MAX_STEPS=1000 BATCH_SIZE=8 sbatch run_scripts/train_libero_cosmos2.sh
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `configs/libero.yaml` | YAML config file |
| `--dataset_path` | (required) | Path to LeRobot-format dataset |
| `--max_steps` | 50000 | Number of training steps |
| `--per_device_train_batch_size` | 4 | Batch size per GPU |
| `--gradient_accumulation_steps` | 8 | Gradient accumulation steps |
| `--freeze_video_backbone` | true | Freeze Cosmos backbone (train only 73M head params) |
| `--lora_rank` | 0 | LoRA rank (0 = disabled) |
| `--report_to` | wandb | Logging: `wandb`, `tensorboard`, `none` |

## Evaluation

### Option 1: Server/Client (Recommended)

Use an HTTP server when the GPU environment (mimic_video) and LIBERO environment (libero) are separate.

**Terminal 1 — Policy server (mimic_video env, GPU)**
```bash
conda activate mimic_video
python scripts/serve_policy.py \
    --checkpoint_path ./checkpoints/libero/checkpoint-60000 \
    --port 5555
```

**Terminal 2 — LIBERO eval client (libero env, CPU)**
```bash
conda activate libero
python scripts/eval_libero_client.py \
    --task_suite_name libero_spatial \
    --task_idx 0 \
    --port 5555 \
    --video_out_path ./eval_output/libero_spatial
```

The client only requires Python stdlib + numpy + imageio, so no extra installation is needed in the libero environment.

### Option 2: Single-process

```bash
python scripts/eval_libero.py \
    --checkpoint_path ./checkpoints/libero/checkpoint-60000 \
    --task_suite_name libero_spatial \
    --num_trials_per_task 50
```

### Eval Output

```
eval_output/
├── rollout_{task}_{episode}_{success/failure}.mp4   # Episode rollout video
├── rollout_{task}_ep00_img.png                      # First episode front image
├── rollout_{task}_ep00_wrist.png                    # First episode wrist image
└── {task_idx}_results.txt                           # Per-task success rate
```

## Tests

```bash
# Unit tests (CPU only, no model download required)
pytest tests/test_data_pipeline.py -v
pytest tests/test_training.py -v
pytest tests/test_policy.py -v

# All unit tests
pytest tests/test_data_pipeline.py tests/test_training.py tests/test_policy.py -v

# GPU integration tests (requires Cosmos model weights + real data)
python tests/gpu_test_g4_cosmos2_frozen.py    # Cosmos 2.0 frozen backbone training
python tests/gpu_test_g5_normalized.py        # Full pipeline with normalization
python tests/gpu_test_safetensors.py          # Checkpoint save/load
```

## Project Structure

```
mimic-video/
├── configs/
│   └── libero.yaml                 # LIBERO training config
├── mimic_video/
│   ├── mimic_video.py              # MimicVideo model (DiT action head + flow matching)
│   ├── cosmos_predict.py           # Cosmos 1.0/2.0 video wrapper
│   ├── policy.py                   # MimicVideoPolicy (obs → action inference)
│   ├── data/
│   │   ├── data_config.py          # BaseDataConfig + LiberoDataConfig
│   │   ├── dataset.py              # LeRobot-format dataset loader
│   │   ├── transforms.py           # Video/state/action transforms
│   │   └── normalization.py        # Normalization utilities
│   ├── training/
│   │   ├── trainer.py              # MimicVideoTrainer (HF Trainer subclass)
│   │   ├── runner.py               # TrainRunner (training orchestration)
│   │   └── collator.py             # Data collator
│   └── eval/
│       ├── libero_eval.py          # LIBERO environment eval loop
│       ├── libero_utils.py         # Env setup, image preprocessing, action conversion
│       └── server.py               # HTTP policy server/client
├── scripts/
│   ├── train.py                    # Training entry point
│   ├── eval_libero.py              # Eval CLI (direct / server mode)
│   ├── eval_libero_client.py       # HTTP client eval (for libero env)
│   ├── serve_policy.py             # Policy HTTP server
│   └── compute_stats.py            # Dataset normalization statistics
├── run_scripts/
│   ├── train_libero_cosmos2.sh     # SLURM training script
│   └── eval_libero_cosmos2.sh      # SLURM evaluation script
└── tests/
    ├── test_data_pipeline.py       # Data pipeline unit tests
    ├── test_training.py            # Training unit tests
    ├── test_policy.py              # Policy inference unit tests
    └── gpu_test_*.py               # GPU integration tests
```

## Citations

```bibtex
@inproceedings{Pai2025mimicvideoVM,
    title   = {mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs},
    author  = {Jonas Pai and Liam Achenbach and Victoriano Montesinos and Benedek Forrai and Oier Mees and Elvis Nava},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:283920528}
}
```

```bibtex
@misc{kim2026cosmospolicyfinetuningvideo,
    title   = {Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning},
    author  = {Moo Jin Kim and Yihuai Gao and Tsung-Yi Lin and Yen-Chen Lin and Yunhao Ge and Grace Lam and Percy Liang and Shuran Song and Ming-Yu Liu and Chelsea Finn and Jinwei Gu},
    year    = {2026},
    eprint  = {2601.16163},
    archivePrefix = {arXiv},
}
```
