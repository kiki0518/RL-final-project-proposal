#!/bin/bash
#SBATCH --account=MST114564
#SBATCH --partition=normal
#SBATCH --job-name=grpo4IDC
#SBATCH --nodes=1 
#SBATCH --ntasks=1                    # 注意：SLURM 的 ntasks 指的是 job 啟動次數
#SBATCH --gres=gpu:1                  # 重要：請求 2 張 GPU
#SBATCH --cpus-per-task=8             # 增加 CPU 核心，因為你有 2 個進程，每個配 4 核
#SBATCH --mem=64G                     # 記憶體也建議加倍，避免 OOM
#SBATCH --time=24:00:00 
#SBATCH --output=log/grpo_train_%j.out 
#SBATCH --error=log/grpo_train_%j.err

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

# 2. 設定 Java 路径（確保 CIDEr 評分器可運作）
export JAVA_HOME=$CONDA_PREFIX
export PATH=$JAVA_HOME/bin:$PATH

torchrun --nproc_per_node=1 grpo_mini.py \
    --do_train \
    --datatype spot \
    --data_path ${DATA_PATH} \
    --features_path ${DATA_PATH}/data/resized_images/ \
    --output_dir ./ckpts/grpo_mini \
    --max_words 20
