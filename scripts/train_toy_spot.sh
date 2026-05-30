#!/bin/bash

DATA_PATH=toy-spot
GT_PATH=toy-gt

python -m torch.distributed.launch --nproc_per_node=1 --master_port=5574 main_task_caption.py \
--do_train \
--num_thread_reader=4 \
--epochs=5 \
--batch_size=16 \
--n_display=10 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/images \
--output_dir ckpts/ckpt_toy_spot_caption \
--lr 1e-4 \
--max_words 8 \
--batch_size_val 16 \
--datatype spot \
--gt_dir ${GT_PATH} \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--linear_patch 2d \
--pretrained_clip_name ViT-B/32 \
--init_model ckpts/pretrained/pytorch_model.bin.spot
