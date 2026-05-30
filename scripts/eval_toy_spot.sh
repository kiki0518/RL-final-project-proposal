#!/bin/bash

DATA_PATH=toy-spot
GT_PATH=toy-gt
MODEL_PATH=${1:-ckpts/trained/pytorch_model.bin.spot}

python -m torch.distributed.launch --nproc_per_node=1 --master_port=5575 main_task_caption.py \
--do_eval \
--num_thread_reader=4 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/images \
--output_dir ckpts/ckpt_toy_spot_eval \
--batch_size_val 16 \
--max_words 8 \
--datatype spot \
--gt_dir ${GT_PATH} \
--linear_patch 2d \
--pretrained_clip_name ViT-B/32 \
--init_model ${MODEL_PATH}
