#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}"
PROMPT="${PROMPT:-Picture 1 is the before image and Picture 2 is the after image. Write a short factual caption describing all visible differences between the two images. If nothing changed, write: no visible change. Output only the caption.}"
ADAPTER_PATH="${ADAPTER_PATH:?Set ADAPTER_PATH to the trained adapter directory}"
DATA_JSON="${DATA_JSON:-spot-the-diff/reformat_val.json}"
GT_JSON="${GT_JSON:-gt/spot_total_change_captions_reformat.json}"
OUT_DIR="${OUT_DIR:-outputs/qwen25_vl_spot_qlora_eval}"

python vlm/qwen25_vl_spot_eval.py \
  --model_name "${MODEL_NAME}" \
  --data_json "${DATA_JSON}" \
  --image_dir spot-the-diff/images \
  --output_json "${OUT_DIR}/hyp.json" \
  --prompt "${PROMPT}" \
  --max_new_tokens 64 \
  --torch_dtype float16 \
  --attn_implementation sdpa \
  --adapter_path "${ADAPTER_PATH}"

python vlm/spot_caption_metrics.py \
  --gt_json "${GT_JSON}" \
  --pred_json "${OUT_DIR}/hyp.json" \
  --output_json "${OUT_DIR}/metrics.json"
