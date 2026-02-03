#!/bin/bash
#SBATCH --job-name="MimicVideo-Cosmos2-LIBERO"
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-mimic_video_libero.out
#SBATCH --error=out/%j-mimic_video_libero.err

# ---- Paths ----
WORK_DIR=/rlwrld3/home/hojin/vam_workspace/mimic-video
CONDA_PATH=/rlwrld3/home/hojin/miniconda3
DATA_DIR=/sjw_alinlab2/home/myungkyu/workspace/AlinVLA/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta
CKPT_DIR=$WORK_DIR/checkpoints/libero_cosmos2_frozen

# ---- Environment ----
export WANDB_PROJECT=mimic-video-libero
source $CONDA_PATH/bin/activate mimic_video

# ---- Training ----
NUM_GPUS=2
MAX_STEPS=${MAX_STEPS:-60000}
SAVE_STEPS=${SAVE_STEPS:-10000}
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-1}
RUN_NAME=${RUN_NAME:-cosmos2_frozen_libero}

cd $WORK_DIR

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node=$NUM_GPUS scripts/train.py \
        --config configs/libero.yaml \
        --dataset_path $DATA_DIR \
        --output_dir $CKPT_DIR \
        --max_steps $MAX_STEPS \
        --save_steps $SAVE_STEPS \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --run_name $RUN_NAME \
        --freeze_video_backbone \
        --report_to wandb
else
    python scripts/train.py \
        --config configs/libero.yaml \
        --dataset_path $DATA_DIR \
        --output_dir $CKPT_DIR \
        --max_steps $MAX_STEPS \
        --save_steps $SAVE_STEPS \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --run_name $RUN_NAME \
        --freeze_video_backbone \
        --report_to wandb
fi
