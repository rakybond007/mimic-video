#!/bin/bash
#SBATCH --job-name="MimicVideo-Eval-8f"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --array=0-9
#SBATCH --output=out/%A_%a-mimic_eval_8f.out
#SBATCH --error=out/%A_%a-mimic_eval_8f.err

# ---- Paths ----
WORK_DIR=/rlwrld3/home/hojin/vam_workspace/mimic-video
BASE_DIR=/rlwrld3/home/hojin/multigpu_workspace
CONDA_PATH=/rlwrld3/home/hojin/miniconda3

# ---- Checkpoint ----
CKPT_NAME="libero_cosmos2_frozen_8frame_60k"
CKPT_STEP=30000
CKPT_DIR=$WORK_DIR/checkpoints/$CKPT_NAME/checkpoint-$CKPT_STEP
NUM_FRAMES=8
NUM_TRIALS=3
OUTPUT_BASE=$WORK_DIR/eval_output/${CKPT_NAME}_${CKPT_STEP}

# ---- Setup LIBERO config to avoid interactive input prompt ----
LIBERO_CONFIG_DIR="$BASE_DIR/.libero"
LIBERO_ROOT="$BASE_DIR/LIBERO/libero/libero"
export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_DIR"
mkdir -p "$LIBERO_CONFIG_DIR"
# Always overwrite to ensure correct paths
cat > "$LIBERO_CONFIG_DIR/config.yaml" <<EOF
benchmark_root: $LIBERO_ROOT
bddl_files: $LIBERO_ROOT/bddl_files
init_states: $LIBERO_ROOT/init_files
datasets: $BASE_DIR/LIBERO/libero/datasets
assets: $LIBERO_ROOT/assets
EOF

# ---- Port (unique per array task) ----
PORT=$((5555 + ${SLURM_ARRAY_TASK_ID:-0}))
TASK_IDX=${SLURM_ARRAY_TASK_ID:-0}

echo "[i] Evaluating: CKPT=$CKPT_DIR, NUM_FRAMES=$NUM_FRAMES, NUM_TRIALS=$NUM_TRIALS, TASK_IDX=$TASK_IDX, PORT=$PORT"

# ---- Start policy server (mimic_video env) ----
"$CONDA_PATH"/envs/mimic_video/bin/python "$WORK_DIR/scripts/serve_policy.py" \
    --checkpoint_path "$CKPT_DIR" \
    --port $PORT &
SERVE_PID=$!

# Wait for server to load model (~85s for Cosmos2, client retries if not ready)
sleep 90

# ---- Run evaluation for each task suite (libero env) ----
TASK_NAMES=("libero_10" "libero_goal" "libero_object" "libero_spatial")
EVAL_PIDS=()

for TASK_NAME in "${TASK_NAMES[@]}"; do
    OUTPUT_DIR="$OUTPUT_BASE/$TASK_NAME"
    mkdir -p "$OUTPUT_DIR"
    "$CONDA_PATH"/envs/libero/bin/python "$WORK_DIR/scripts/eval_libero_client.py" \
        --task_suite_name "$TASK_NAME" \
        --video_out_path "$OUTPUT_DIR" \
        --task_idx "$TASK_IDX" \
        --port $PORT \
        --replan_steps 5 \
        --num_frames $NUM_FRAMES \
        --num_trials_per_task $NUM_TRIALS \
        >& "$OUTPUT_DIR/eval-$TASK_IDX.log" &
    EVAL_PIDS+=($!)
done

# Wait for all eval processes to complete
for pid in "${EVAL_PIDS[@]}"; do
    wait "$pid"
done

# Kill server
kill "$SERVE_PID"
echo "[i] Finished TASK_IDX=$TASK_IDX."
