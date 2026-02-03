#!/bin/bash
#SBATCH --job-name="MimicVideo-Eval-LIBERO"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=sjw_alinlab
#SBATCH --array=0-9
#SBATCH --output=out/%A_%a-mimic_eval.out
#SBATCH --error=out/%A_%a-mimic_eval.err

# ---- Paths ----
WORK_DIR=/rlwrld3/home/hojin/vam_workspace/mimic-video
BASE_DIR=/rlwrld3/home/hojin/multigpu_workspace
CONDA_PATH=/rlwrld3/home/hojin/miniconda3

# ---- Checkpoint ----
CKPT_DIR=$WORK_DIR/checkpoints/test_nan_fix/checkpoint-100
OUTPUT_BASE=$WORK_DIR/eval_output

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

echo "[i] Evaluating: CKPT=$CKPT_DIR, TASK_IDX=$TASK_IDX, PORT=$PORT"

# ---- Start policy server (mimic_video env) ----
"$CONDA_PATH"/envs/mimic_video/bin/python "$WORK_DIR/scripts/serve_policy.py" \
    --checkpoint_path "$CKPT_DIR" \
    --port $PORT &
SERVE_PID=$!

# Wait for server to load model (~48s for Cosmos2 7B)
sleep 60

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
