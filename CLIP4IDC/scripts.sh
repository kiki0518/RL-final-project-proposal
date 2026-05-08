#!/bin/bash
#SBATCH --account=MST114564
#SBATCH --partition=normal
#SBATCH --job-name=CLIP4IDC_train     # 改個符合專案的名字
#SBATCH --nodes=1 
#SBATCH --ntasks=1                    # 注意：SLURM 的 ntasks 指的是 job 啟動次數
#SBATCH --gres=gpu:1                  # 重要：請求 2 張 GPU
#SBATCH --cpus-per-task=8             # 增加 CPU 核心，因為你有 2 個進程，每個配 4 核
#SBATCH --mem=64G                     # 記憶體也建議加倍，避免 OOM
#SBATCH --time=24:00:00 
#SBATCH --output=log/clip_train_%j.out 
#SBATCH --error=log/clip_train_%j.err

source ~/miniconda3/etc/profile.d/conda.sh

conda activate IDC_v3

cd /work/sixplus7/IDC/CLIP4IDC/

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo,eth0,en,em
# export NCCL_DEBUG=INFO
export TORCH_CUDNN_V8_API_ENABLED=1
export torch_cudnn_benchmark=0
export torch_cudnn_enabled=0
export CUDA_LAUNCH_BLOCKING=1

DATA_PATH=spot-the-diff
torchrun --nproc_per_node=1 ppo_idc_trainer.py \
--do_train \
--num_thread_reader=4 \
--epochs=20 \
--batch_size=128 \
--n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/data/resized_images/ \
--output_dir ckpts/ckpt_spot_retrieval \
--lr 1e-4 \
--max_words 32 \
--batch_size_val 128 \
--datatype spot \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--linear_patch 2d \
--pretrained_clip_name ViT-B/32

