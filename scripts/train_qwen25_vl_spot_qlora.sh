#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}"
PROMPT="${PROMPT:-Picture 1 is the before image and Picture 2 is the after image. Write a short factual caption describing all visible differences between the two images. If nothing changed, write: no visible change. Output only the caption.}"
PREP_DIR="${PREP_DIR:-prepared/qwen25_vl_spot_sft}"
OUT_DIR="${OUT_DIR:-outputs/qwen25_vl_spot_qlora}"

python vlm/prepare_spot_sft.py \
  --train_json spot-the-diff/reformat_train.json \
  --val_json spot-the-diff/reformat_val.json \
  --image_dir spot-the-diff/images \
  --output_dir "${PREP_DIR}" \
  --prompt "${PROMPT}"

python vlm/train_qwen25_vl_spot_qlora.py \
  --model_name "${MODEL_NAME}" \
  --train_jsonl "${PREP_DIR}/train.jsonl" \
  --val_jsonl "${PREP_DIR}/val.jsonl" \
  --output_dir "${OUT_DIR}" \
  --torch_dtype bfloat16 \
  --attn_implementation sdpa \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --logging_steps 10
