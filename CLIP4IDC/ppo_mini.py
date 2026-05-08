import os
import torch
import torch.nn as nn
from torch.distributions import Categorical
import wandb
import numpy as np
from modules.modeling import CLIP4IDC
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from dataloaders.data_dataloaders import DATALOADER_DICT
from pycocoevalcap.cider.cider import Cider
from main_task_caption import get_args

# 設定 Java 路徑 (請確保這指向你的 Conda 環境)
os.environ['JAVA_HOME'] = os.environ.get('CONDA_PREFIX', '/usr/lib/jvm/default-java')

def main():
    # 1. 初始化參數與 W&B
    args = get_args()
    args.batch_size = 8 # 小規模訓練，防止顯存爆炸
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ClipTokenizer()
    
    wandb.init(project="IDC-PPO-Mini", name="real_ppo_experiment")

    # 2. 載入 SFT 預訓練模型
    print("Loading SFT model from ckpts/ckpt_spot_caption/pytorch_model.bin.0 ...")
    checkpoint = torch.load("ckpts/ckpt_spot_caption/pytorch_model.bin.0", map_location='cpu')
    model = CLIP4IDC.from_pretrained("cross-base", "decoder-base", state_dict=checkpoint, task_config=args)
    
    # 3. 強制注入 Critic Head 與 PPO 採樣邏輯
    model.critic_head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1)).to(device)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    cider_scorer = Cider()

    # 4. 準備資料流
    train_dataloader, _, _ = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)

    print("Starting Mini PPO Training...")
    for epoch in range(5): # 跑 5 個 Epoch
        for step, batch in enumerate(train_dataloader):
            if step > 20: break # 每個 Epoch 只跑 20 步，達成「小規模」需求

            # 解析資料
            input_ids, input_mask, segment_ids, bef_img, aft_img, img_mask, \
            _, _, gt_ids = [t.to(device) for t in batch[:-1]]
            image_pair = torch.cat([bef_img, aft_img], 1)

            # --- 第一階段：Rollout (採樣生成) ---
            model.eval()
            with torch.no_grad():
                _, _, _, visual_output = model.get_sequence_visual_output(
                    input_ids, segment_ids, input_mask, image_pair, img_mask
                )
                max_len = 20
                curr_input = torch.zeros((image_pair.size(0), max_len), device=device).long()
                curr_input[:, 0] = tokenizer.vocab["<|startoftext|>"]
                curr_mask = torch.zeros((image_pair.size(0), max_len), device=device).long()
                curr_mask[:, 0] = 1 # 第一個字是 Start，必須被看見       




         # 簡單採樣邏輯：這裡模擬從 Decoder 取樣 (Loop 版)
                # curr_input = torch.full((image_pair.size(0), 1), tokenizer.vocab["<|startoftext|>"], device=device).long()
                # curr_mask = torch.ones_like(curr_input)
                sampled_ids = []
                log_probs_list = []

                live_mask = torch.ones_like(curr_input)                

                for _ in range(args.max_words-1):
                    logits = model.decoder_caption(visual_output, img_mask, curr_input, curr_mask, shaped=True, get_logits=True)
                    next_token_logits = logits[:, -1, :]
                    dist = Categorical(logits=next_token_logits)
                    action = dist.sample()
                    
                    sampled_ids.append(action)
                    log_probs_list.append(dist.log_prob(action))
                    
                    curr_input[:, i+1] = action
                    curr_mask[:, i+1] = 1
                    # curr_input = torch.cat([curr_input, action.unsqueeze(1)], dim=1)

                sampled_ids_tensor = torch.stack(sampled_ids, dim=1)
                old_log_probs = torch.stack(log_probs_list, dim=1).sum(dim=-1).detach()
                state_val = model.critic_head(visual_output.mean(dim=1)).squeeze(-1).detach()

            # --- 第二階段：算真分 (CIDEr Reward) ---
            res = {i: [tokenizer.decode(g.tolist())] for i, g in enumerate(sampled_ids_tensor)}
            gts = {i: [tokenizer.decode(gt.tolist())] for i, gt in enumerate(gt_ids)}
            _, cider_scores = cider_scorer.compute_score(gts, res)
            rewards = torch.tensor(cider_scores, device=device).float()

            # --- 第三階段：PPO 更新 ---
            model.train()
            for _ in range(2): # PPO 內迴圈
                # 重新計算 current policy
                new_logits = model.decoder_caption(visual_output, img_mask, curr_input[:, :-1], curr_mask[:, :-1], shaped=True, get_logits=True)
                new_dist = Categorical(logits=new_logits)
                new_log_probs = new_dist.log_prob(sampled_ids_tensor).sum(dim=-1)
                entropy = new_dist.entropy().mean()
                current_val = model.critic_head(visual_output.mean(dim=1)).squeeze(-1)

                # PPO 損失
                ratio = torch.exp(new_log_probs - old_log_probs)
                advantage = rewards - state_val
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * nn.MSELoss()(current_val, rewards)
                loss = actor_loss + critic_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 紀錄數據
            wandb.log({
                "reward": rewards.mean().item(),
                "loss": loss.item(),
                "sample": wandb.Html(tokenizer.decode(sampled_ids_tensor[0].tolist()))
            })
            print(f"Epoch {epoch} Step {step} | Reward: {rewards.mean().item():.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()
