import os
import torch
import torch.nn as nn
from torch.distributions import Categorical
import wandb
import argparse
from modules.modeling import CLIP4IDC
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from dataloaders.data_dataloaders import DATALOADER_DICT
from pycocoevalcap.cider.cider import Cider

# 設定 Java 路徑確保 Cider 能跑
os.environ['JAVA_HOME'] = '/usr/lib/jvm/default-java' # 依伺服器環境調整

def get_ppo_args():
    # 這裡可以沿用你原本 get_args 的邏輯，再加上 PPO 專用參數
    from main_task_caption import get_args
    args = get_args()
    args.ppo_lr = 1e-6
    args.eps_clip = 0.2
    args.ppo_epochs = 4
    return args

def train_ppo_real_step(model, tokenizer, batch, device, cider_scorer, optimizer):
    # --- 1. 資料準備 ---
    input_ids, input_mask, segment_ids, bef_img, aft_img, img_mask, \
    _, _, gt_ids = [t.to(device) for t in batch[:-1]]
    
    # --- 2. Rollout: 採樣生成 ---
    model.eval()
    with torch.no_grad():
        # 取得影像特徵 (State)
        _, _, _, visual_output = model.get_sequence_visual_output(
            input_ids, segment_ids, input_mask, torch.cat([bef_img, aft_img], 1), img_mask
        )
        # 這裡需自定義一個 sample_logic 函數來進行隨機採樣
        sample_ids, old_log_probs = model.sample_logic(visual_output) 
        # 此處簡化邏輯供參考
  

    # --- 3. 計算 Reward (CIDEr) ---
    res = {i: [tokenizer.decode(g)] for i, g in enumerate(sample_ids.tolist())}
    gts = {i: [tokenizer.decode(gt)] for i, gt in enumerate(gt_ids.tolist())}
    _, cider_scores = cider_scorer.compute_score(gts, res)
    rewards = torch.tensor(cider_scores, device=device).float()

    # --- 4. PPO Update ---
    model.train()
    for _ in range(4): 
        # 這裡執行 Actor-Critic 更新邏輯
        # ≈loss = ...
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        pass

    return rewards.mean().item()

def main():
    args = get_ppo_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ClipTokenizer()
    
    wandb.init(project="IDC-PPO-Project", name="ppo_real_test")

    # 載入 SFT 練好的權重
    checkpoint = torch.load("ckpts/ckpt_spot_caption/pytorch_model.bin.0", map_location='cpu')
    model = CLIP4IDC.from_pretrained("cross-base", "decoder-base", state_dict=checkpoint, task_config=args)
    
    # 加入 Critic Head
    model.critic_head = nn.Linear(768, 1).to(device)
    model.to(device)

    # 準備 Scorer 與 Optimizer
    cider_scorer = Cider()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.ppo_lr)

    train_dataloader, _, _ = DATALOADER_DICT["spot"]["train"](args, tokenizer)

    for epoch in range(args.epochs):
        for step, batch in enumerate(train_dataloader):
            reward = train_ppo_real_step(model, tokenizer, batch, device, cider_scorer, optimizer)
            
            wandb.log({"step_reward": reward})
            if step % 10 == 0:
                print(f"Epoch {epoch} Step {step} Reward: {reward}")
            
            # Toy task 建議限制步數
            if step > 100: break 

if __name__ == "__main__":
    main()
