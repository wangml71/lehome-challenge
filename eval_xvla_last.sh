#!/usr/bin/env bash
set -euo pipefail

# CKPT_DIR="${CKPT_DIR:-/home/wangml71/lehome-challenge/lehome-challenge/outputs/train/xvla_four_types/checkpoints/20260402_050000}"
CKPT_DIR="${CKPT_DIR:-/home/wangml71/lehome-challenge/lehome-challenge/outputs/train/xvla_folding/checkpoints/50000}"
# CKPT_DIR="${CKPT_DIR:-/home/wangml71/lehome-challenge/lehome-challenge/outputs/train/xvla_base_fold/20260407/pretrained_model}"
# CKPT_DIR="${CKPT_DIR:-/home/wangml71/lehome-challenge/lehome-challenge/outputs/train/multi_task_dit/checkpoints/030000/pretrained_model}"
# CKPT_DIR="${CKPT_DIR:-/home/wangml71/lehome-challenge/lehome-challenge/outputs/train/wall_oss/20260409/pretrained_model}"
POLICY_PATH="${POLICY_PATH:-}"
DATASET_ROOT="${DATASET_ROOT:-Datasets/example/top_short_merged}"

TASK="${TASK:-LeHome-BiSO101-Direct-Garment-v2}"
GARMENT_TYPE="${GARMENT_TYPE:-}"
NUM_EPISODES="${NUM_EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-600}"
STEP_HZ="${STEP_HZ:-120}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-fold the garment on the table}"

DEVICE="${DEVICE:-cpu}"
HEADLESS="${HEADLESS:-0}"
ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"

SAVE_VIDEO="${SAVE_VIDEO:-0}"
VIDEO_DIR="${VIDEO_DIR:-outputs/eval_videos}"

SAVE_DATASETS="${SAVE_DATASETS:-0}"
EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-Datasets/eval}"

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "CKPT_DIR not found: $CKPT_DIR" >&2
  exit 1
fi

if [[ -z "${POLICY_PATH:-}" ]]; then
  if [[ -e "$CKPT_DIR/config.json" ]]; then
    POLICY_PATH="$CKPT_DIR"
  elif [[ -d "$CKPT_DIR/pretrained_model" ]]; then
    POLICY_PATH="$CKPT_DIR/pretrained_model"
  else
    echo "Cannot infer POLICY_PATH from CKPT_DIR: $CKPT_DIR" >&2
    echo "Set POLICY_PATH to a directory containing config.json and model weights." >&2
    exit 1
  fi
fi

if [[ ! -e "$POLICY_PATH/config.json" && ! -e "$POLICY_PATH/train_config.json" ]]; then
  echo "No config.json or train_config.json found under: $POLICY_PATH" >&2
  exit 1
fi

if [[ -z "${DATASET_ROOT:-}" || ! -d "$DATASET_ROOT" ]]; then
  echo "DATASET_ROOT not found: $DATASET_ROOT" >&2
  exit 1
fi

GARMENT_TYPES_DEFAULT=("top_short" "top_long" "pant_short" "pant_long")
if [[ -z "${GARMENT_TYPE:-}" ]]; then
  GARMENT_TYPES=("${GARMENT_TYPES_DEFAULT[@]}")
else
  IFS=' ' read -r -a GARMENT_TYPES <<<"$GARMENT_TYPE"
fi

if [[ "${#GARMENT_TYPES[@]}" -eq 0 ]]; then
  echo "No garment types specified via GARMENT_TYPE and no defaults available." >&2
  exit 1
fi

args=(
  python -m scripts.eval
  --policy_type lerobot
  --policy_path "$POLICY_PATH"
  --dataset_root "$DATASET_ROOT"
  --task "$TASK"
  --num_episodes "$NUM_EPISODES"
  --max_steps "$MAX_STEPS"
  --step_hz "$STEP_HZ"
  --task_description "$TASK_DESCRIPTION"
  --device "$DEVICE"
)

if [[ "$HEADLESS" == "1" ]]; then
  args+=(--headless)
fi

if [[ "$ENABLE_CAMERAS" == "1" ]]; then
  args+=(--enable_cameras)
fi

if [[ "$SAVE_VIDEO" == "1" ]]; then
  args+=(--save_video --video_dir "$VIDEO_DIR")
fi

if [[ "$SAVE_DATASETS" == "1" ]]; then
  args+=(--save_datasets --eval_dataset_path "$EVAL_DATASET_PATH")
fi

for gt in "${GARMENT_TYPES[@]}"; do
  echo "Evaluating garment_type=$gt"
  "${args[@]}" --garment_type "$gt"
done
