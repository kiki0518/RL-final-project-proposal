# CLIP4IDC

## Setup

```bash
conda create -n idc python=3.8 -y
conda activate idc

conda install -y -c pytorch pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0
pip install -r requirements.txt
```

## Data

Use the Spot-the-Diff data under:

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

## Weights

```bash
mkdir -p ckpts/pretrained ckpts/trained

wget -O modules/ViT-B-32.pt \
https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt

python -m pip install gdown

gdown --folder "https://drive.google.com/drive/folders/1qOYVpZy57clJPF6AThsnO0Tfy4zq-gg1" -O ckpts/pretrained
gdown --folder "https://drive.google.com/drive/folders/18UfIvwKt0EE14EbogJycMmANpUJtsZbE" -O ckpts/trained
```

## Spot-the-Diff Evaluation

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

Outputs:

```text
ckpts/ckpt_spot_caption_eval/log.txt
ckpts/ckpt_spot_caption_eval/hyp_ep_eval.json
```

## Spot-the-Diff Retrieval Training

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=6666 main_task_retrieval.py \
  --do_train \
  --num_thread_reader=4 \
  --epochs=20 \
  --batch_size=32 \
  --n_display=50 \
  --data_path "$DATA_PATH" \
  --features_path "$DATA_PATH/images" \
  --output_dir ckpts/ckpt_spot_retrieval \
  --lr 1e-4 \
  --max_words 32 \
  --batch_size_val 32 \
  --datatype spot \
  --coef_lr 1e-3 \
  --freeze_layer_num 0 \
  --linear_patch 2d \
  --pretrained_clip_name ViT-B/32
```

## Spot-the-Diff Caption Training

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=5564 main_task_caption.py \
  --do_train \
  --num_thread_reader=4 \
  --epochs=50 \
  --batch_size=16 \
  --n_display=50 \
  --data_path "$DATA_PATH" \
  --features_path "$DATA_PATH/images" \
  --output_dir ckpts/ckpt_spot_caption \
  --lr 1e-4 \
  --max_words 32 \
  --batch_size_val 32 \
  --datatype spot \
  --coef_lr 1e-3 \
  --freeze_layer_num 0 \
  --linear_patch 2d \
  --pretrained_clip_name ViT-B/32 \
  --init_model ckpts/pretrained/pytorch_model.bin.spot
```

Original upstream instructions are in `original_README.md`.
