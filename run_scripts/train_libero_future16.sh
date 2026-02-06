#!/bin/bash
#SBATCH --job-name="MimicVideo-Future16"
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=rlwrld
#SBATCH --output=out/%j-mimic_video_future16.out
#SBATCH --error=out/%j-mimic_video_future16.err

# ---- Paths ----
WORK_DIR=/rlwrld3/home/hojin/vam_workspace/mimic-video
CONDA_PATH=/rlwrld3/home/hojin/miniconda3
DATA_DIR=/sjw_alinlab2/home/myungkyu/workspace/AlinVLA/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta
CKPT_DIR=$WORK_DIR/checkpoints/libero_cosmos2_past8_future1_idx16

# ---- Environment ----
export WANDB_PROJECT=mimic-video-libero
source $CONDA_PATH/bin/activate mimic_video

# ---- Training ----
NUM_GPUS=2
MAX_STEPS=60000
SAVE_STEPS=10000
BATCH_SIZE=16
GRAD_ACCUM=1
NUM_FRAMES=8
NUM_FUTURE_FRAMES=1
FUTURE_FRAME_START=16
RUN_NAME=cosmos2_frozen_lang_past8_future1_idx16

cd $WORK_DIR

MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT scripts/train.py \
    --config configs/libero.yaml \
    --dataset_path $DATA_DIR \
    --output_dir $CKPT_DIR \
    --max_steps $MAX_STEPS \
    --save_steps $SAVE_STEPS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --run_name $RUN_NAME \
    --num_frames $NUM_FRAMES \
    --num_future_frames $NUM_FUTURE_FRAMES \
    --future_frame_start $FUTURE_FRAME_START \
    --freeze_video_backbone \
    --inject_language_tokens \
    --report_to wandb
