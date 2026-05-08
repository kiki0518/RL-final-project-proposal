import os
import torch
import torch.nn as nn
from torch.distributions import Categorical
import wandb
from modules.modeling import CLIP4IDC
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from dataloaders.data_dataloaders import DATALOADER_DICT
from pycocoevalcap.cider.cider import Cider
from main_task_caption import get_args

# 環境設定
os.environ['JAVA_HOME'] = os.environ.get('CONDA_PREFIX', '/usr/lib/jvm/default-java')

def run_generate_sample(model, visual_output, img_mask, max_len, tokenizer, device):
    """ 負責隨機採樣生成句子，並回傳 log_probs """
    batch_size = visual_output.size(0)
    # 這裡固定長度為 100，避免維度報錯
    FIXED_LEN = 77
    
    curr_input = torch.zeros((batch_size, FIXED_LEN), device=device).long()
    curr_input[:, 0] = tokenizer.vocab["<|startoftext|>"]
    curr_mask = torch.zeros((batch_size, FIXED_LEN), device=device).long()
    curr_mask[:, 0] = 1 

    sampled_ids = []
    log_probs_list = []

    for i in range(FIXED_LEN - 1):
        # 取得 logits
        logits = model.decoder_caption(visual_output, img_mask, curr_input, curr_mask, shaped=True, get_logits=True)
        
        # 隨機採樣
        next_token_logits = logits[:, i, :] 
        dist = Categorical(logits=next_token_logits)
        action = dist.sample()
        
        # 填入下一格
        curr_input[:, i+1] = action
        curr_mask[:, i+1] = 1
        
        sampled_ids.append(action)
        log_probs_list.append(dist.log_prob(action))
        
        # 如果所有 batch 都抽到結束符，可以提早結束 (選配)
        # if (action == tokenizer.vocab["<|endoftext|>"]).all(): break

    return torch.stack(sampled_ids, dim=1), torch.stack(log_probs_list, dim=1).sum(dim=-1)

def get_log_probs(model, visual_output, img_mask, full_input_ids):
    """ 重新計算給定序列的 log_probs (用於更新階段) """
    # 假設 full_input_ids 長度是 100
    batch_size = full_input_ids.size(0)
    # 建立對應的 mask
    full_mask = (full_input_ids != 0).long() 
    
    logits = model.decoder_caption(visual_output, img_mask, full_input_ids, full_mask, shaped=True, get_logits=True)
    
    # 計算每個位置的 log_prob
    # logits 對應預測下一個字，所以偏移一位
    dist = Categorical(logits=logits[:, :-1, :])
    target_ids = full_input_ids[:, 1:]
    log_probs = dist.log_prob(target_ids)
    
    # 只加總非 padding 的部分
    return (log_probs * full_mask[:, 1:]).sum(dim=-1)

def main():
    args = get_args()
    args.batch_size = 2  # GRPO 會對每組圖生成多個句子，batch size 設小一點
    group_size = 8       # GRPO 的關鍵：每一組圖生成 8 個不同的描述 (Group Size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ClipTokenizer()
    
    wandb.init(project="IDC-GRPO-Mini", name="grpo_real_experiment")

    # 載入 SFT 模型
    checkpoint = torch.load("ckpts/ckpt_spot_caption/pytorch_model.bin.0", map_location='cpu')
    model = CLIP4IDC.from_pretrained("cross-base", "decoder-base", state_dict=checkpoint, task_config=args)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    cider_scorer = Cider()

    train_dataloader, _, _ = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)

    print(f"Starting GRPO Training (Group Size: {group_size})...")

    for epoch in range(5):
        for step, batch in enumerate(train_dataloader):
            if step > 10: break # 小規模測試

            # 1. 解析資料
            input_ids, input_mask, segment_ids, bef_img, aft_img, img_mask, \
            _, _, gt_ids = [t.to(device) for t in batch[:-1]]
            
            # 2. Rollout: 生成 Group 描述
            model.eval()
            with torch.no_grad():
                _, _, _, visual_output = model.get_sequence_visual_output(
                    input_ids, segment_ids, input_mask, torch.cat([bef_img, aft_img], 1), img_mask
                )
                
                # 對每一張圖重複生成 group_size 次
                # [Batch, Group, SeqLen]
                all_group_ids = []
                all_group_log_probs = []
                
                for _ in range(group_size):
                    # 這裡使用隨機採樣生成一個句子
                    sampled_ids, lp = run_generate_sample(model, visual_output, img_mask, args.max_words, tokenizer, device)
                    # sampled_ids, log_probs = model.generate_sample(visual_output, img_mask, args.max_words)
                    all_group_ids.append(sampled_ids)
                    all_group_log_probs.append(log_probs)

            # 3. 計算 Reward 與 Group Advantage
            # 算分 (CIDEr)
            flat_res = {i: [tokenizer.decode(ids.tolist())] for i, ids in enumerate(torch.cat(all_group_ids))}
            # 複製 Ground Truth 以匹配平坦化後的數量
            repeated_gt = []
            for gt in gt_ids:
                repeated_gt.extend([gt] * group_size)
            flat_gts = {i: [tokenizer.decode(gt.tolist())] for i, gt in enumerate(repeated_gt)}
            
            _, scores = cider_scorer.compute_score(flat_gts, flat_res)
            rewards = torch.tensor(scores, device=device).view(args.batch_size, group_size)

            # GRPO 的核心：組內標準化 (Group Normalization)
            mean_reward = rewards.mean(dim=1, keepdim=True)
            std_reward = rewards.std(dim=1, keepdim=True)
            advantages = (rewards - mean_reward) / (std_reward + 1e-8)
            advantages = advantages.view(-1) # 平坦化用於 Loss 計算

            # 4. GRPO Update
            model.train()
            start_tokens = torch.full((all_group_ids_tensor.size(0), 1), tokenizer.vocab["<|startoftext|>"], device=device)
            full_input_for_update = torch.cat([start_tokens, all_group_ids_tensor], dim=1)

            current_log_probs = get_log_probs(model, visual_output, img_mask.repeat_interleave(group_size, dim=0), full_input_for_update)
            # 重新計算 log_probs (通常會加 KL Divergence 懲罰)
            # current_log_probs = model.get_log_probs(visual_output, img_mask, torch.cat(all_group_ids))
            
            # 簡化版 GRPO Loss: -(ratio * advantages)
            # 這裡 ratio = exp(current - old)
            old_log_probs = torch.cat(all_group_log_probs).detach()
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            # Clipped Surrogate Objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            loss = -torch.min(surr1, surr2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"mean_reward": mean_reward.mean().item(), "loss": loss.item()})
            print(f"Step {step} | Group Mean CIDEr: {mean_reward.mean().item():.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()
