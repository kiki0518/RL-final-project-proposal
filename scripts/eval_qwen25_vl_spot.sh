#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}"
PROMPT="${PROMPT:-Picture 1 is the before image and Picture 2 is the after image. Write a short factual caption describing all visible differences between the two images. If nothing changed, write: no visible change. Output only the caption.}"
OUT_DIR="${OUT_DIR:-outputs/qwen25_vl_spot}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
PROCESSOR_NAME="${PROCESSOR_NAME:-}"
DATA_JSON="${DATA_JSON:-spot-the-diff/reformat_test.json}"
IMAGE_DIR="${IMAGE_DIR:-spot-the-diff/images}"
GT_JSON="${GT_JSON:-gt/spot_total_change_captions_reformat.json}"

CMD=(python vlm/qwen25_vl_spot_eval.py
  --model_name "${MODEL_NAME}"
  --data_json "${DATA_JSON}"
  --image_dir "${IMAGE_DIR}"
  --output_json "${OUT_DIR}/hyp.json"
  --prompt "${PROMPT}"
  --max_new_tokens 64
  --torch_dtype float16
  --attn_implementation sdpa
)

if [[ -n "${ADAPTER_PATH}" ]]; then
  CMD+=(--adapter_path "${ADAPTER_PATH}")
fi

if [[ -n "${PROCESSOR_NAME}" ]]; then
  CMD+=(--processor_name "${PROCESSOR_NAME}")
fi

"${CMD[@]}"

python vlm/spot_caption_metrics.py \
  --gt_json "${GT_JSON}" \
  --pred_json "${OUT_DIR}/hyp.json" \
  --output_json "${OUT_DIR}/metrics.json"
