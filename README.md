# CLIP4IDC

## Setup

Use this for the CLIP commands:

```bash
conda create -n idc python=3.8 -y
conda activate idc

conda install -y -c pytorch pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0
pip install -r requirements.txt
```

Use this for the Qwen commands:

```bash
conda create -n idc-vlm python=3.10 -y
conda activate idc-vlm

conda install -y -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.1
pip install -r requirements.txt
pip install transformers datasets peft trl bitsandbytes accelerate qwen-vl-utils
```

## Data

Put the Spot-the-Diff data at:

```bash
export DATA_PATH="$PWD/spot-the-diff"
```

Expected files:

```text
spot-the-diff/
├── images/
├── train.json
├── val.json
├── test.json
├── reformat_train.json
├── reformat_val.json
└── reformat_test.json
```

Build a scoring file for any split:

```bash
python vlm/build_spot_split_gt.py \
  --split_json spot-the-diff/reformat_val.json \
  --output_json gt/spot_val_change_captions_reformat.json
```

## Weights

```bash
mkdir -p ckpts/pretrained ckpts/trained

wget -O modules/ViT-B-32.pt \
https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt

gdown --folder "https://drive.google.com/drive/folders/1qOYVpZy57clJPF6AThsnO0Tfy4zq-gg1" -O ckpts/pretrained
gdown --folder "https://drive.google.com/drive/folders/18UfIvwKt0EE14EbogJycMmANpUJtsZbE" -O ckpts/trained
```

## Commands

Run these from the repo root.

### CLIP caption eval

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=5564 main_task_caption.py \
  --do_eval \
  --num_thread_reader=4 \
  --data_path "$DATA_PATH" \
  --features_path "$DATA_PATH/images" \
  --output_dir ckpts/ckpt_spot_caption_eval \
  --batch_size_val 32 \
  --datatype spot \
  --freeze_layer_num 0 \
  --linear_patch 2d \
  --pretrained_clip_name ViT-B/32 \
  --init_model ckpts/trained/pytorch_model.bin.spot
```

### CLIP training

```bash
bash scripts/caption_spot.sh
bash scripts/retrieve_spot.sh
```

Set `DATA_PATH` inside those scripts before running them.

### Toy Spot run

```bash
python scripts/create_toy_spot_task.py
bash scripts/train_toy_spot.sh
bash scripts/eval_toy_spot.sh
```

### Qwen base test

```bash
conda activate idc-vlm
bash scripts/eval_qwen25_vl_spot.sh
```

### VLM SFT training

```bash
conda activate idc-vlm
bash scripts/train_qwen25_vl_spot_qlora.sh
```

This saves the trained files to `outputs/qwen25_vl_spot_qlora/final_adapter`.

### VLM SFT testing

```bash
conda activate idc-vlm
ADAPTER_PATH=outputs/qwen25_vl_spot_qlora/final_adapter \
DATA_JSON=spot-the-diff/reformat_test.json \
GT_JSON=gt/spot_total_change_captions_reformat.json \
OUT_DIR=outputs/qwen25_vl_spot_qlora_test \
bash scripts/eval_qwen25_vl_spot_qlora.sh
```

Outputs are saved under `ckpts/` or `outputs/`.

Original upstream instructions are in `original_README.md`.
