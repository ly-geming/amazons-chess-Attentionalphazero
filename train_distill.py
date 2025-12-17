import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
import numpy as np
import os
import sys
import random
import time

sys.path.append(os.getcwd())
from amazons.pytorch.NNet import NNetWrapper
from amazons.train_config import NTo1TrainingConfig

try:
    import amazons_ops
    print("[SUCCESS] C++ amazons_ops module imported successfully!")
except ImportError as e:
    print(f"[ERROR] Cannot import amazons_ops. {e}")
    sys.exit(1)

class AverageMeter(object):
    def __init__(self): self.reset()
    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def apply_consistent_symmetries(boards, srcs, dsts, arrs):
    def idx_to_rc(idx): return idx // 8, idx % 8
    def rc_to_idx(r, c): return r * 8 + c

    if random.random() > 0.5:
        boards = torch.flip(boards, dims=[-1])
        def flip_transform(idx):
            r, c = idx_to_rc(idx)
            return rc_to_idx(r, 7 - c)
        srcs = flip_transform(srcs); dsts = flip_transform(dsts); arrs = flip_transform(arrs)

    k = random.randint(0, 3)
    if k > 0:
        boards = torch.rot90(boards, k, dims=[-2, -1])
        def rot_transform(idx, k_rot):
            r, c = idx_to_rc(idx)
            for _ in range(k_rot): r, c = 7 - c, r
            return rc_to_idx(r, c)
        srcs = rot_transform(srcs, k); dsts = rot_transform(dsts, k); arrs = rot_transform(arrs, k)
        
    return boards, srcs, dsts, arrs

# =========================================================
# 加权准确率 (Weighted Accuracy) 计算
# =========================================================
def calculate_metrics(log_p_move, log_p_arrow, srcs, dsts, arrs, probs):
    # Flatten indices
    move_flat_indices = srcs * 64 + dsts
    
    # Gather probabilities
    log_p_arrow = log_p_arrow.squeeze(1)
    
    current_k = log_p_move.size(0)
    range_idxs = torch.arange(current_k, device=log_p_move.device)

    selected_move_logp = log_p_move[range_idxs, move_flat_indices]
    selected_arrow_logp = log_p_arrow[range_idxs, arrs]
    
    model_logp = selected_move_logp + selected_arrow_logp
    
    # Cross Entropy (Policy Loss Component)
    cross_entropy_sum = -torch.sum(probs * model_logp)
    
    with torch.no_grad():
        # Weighted Accuracy
        pred_move = log_p_move.argmax(dim=1)
        is_match = (pred_move == move_flat_indices)
        weighted_hits = (is_match.float() * probs).sum()
        
        # Acc@5
        _, top5_indices = log_p_move.topk(5, dim=1)
        is_match_5 = (top5_indices == move_flat_indices.unsqueeze(1)).any(dim=1)
        weighted_hits_5 = (is_match_5.float() * probs).sum()
        
        total_prob_mass = probs.sum()

    return cross_entropy_sum, weighted_hits, weighted_hits_5, total_prob_mass

def main():
    # ==========================================
    # Configuration
    # ==========================================
    BATCH_SIZE = 256             
    MAX_ACTIONS_PER_PASS = 16384 
    
    GAMES_PER_LOOP = 100         
    TRAIN_STEPS = 200            
    BUFFER_SIZE = 1000000     
    
    constant_lr = 3e-6
    
    print(f"⚡ Config: Constant LR={constant_lr} | Batch={BATCH_SIZE} Games")
    
    config = NTo1TrainingConfig(
        input_channels=7,
        training_batch_size=BATCH_SIZE,
        learning_rate=constant_lr
    )
    
    DATA_FILE = "teacher_data_joint.bin"
    CHECKPOINT_DIR = "checkpoint"
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)

    class StubGame:
        def getBoardSize(self): return (8, 8)
        def getActionSize(self): return 4096 
    
    nnet_wrapper = NNetWrapper(StubGame(), config)
    device = nnet_wrapper.device
    nnet_wrapper.nnet.train() 
    
    chkpt_path = os.path.join(CHECKPOINT_DIR, "best_joint.pth.tar")
    if os.path.exists(chkpt_path):
        print(f"🔄 Resuming from checkpoint: {chkpt_path}")
        try:
            checkpoint = torch.load(chkpt_path, map_location=device)
            nnet_wrapper.nnet.load_state_dict(checkpoint['state_dict'])
            print("✅ Model weights loaded successfully.")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}. Starting fresh.")
    else:
        print("🆕 No checkpoint found. Starting fresh training.")


    optimizer = optim.AdamW(nnet_wrapper.nnet.parameters(), lr=constant_lr, weight_decay=1e-4)
    scaler = GradScaler()
    
    teacher = amazons_ops.ReplayBuffer(BUFFER_SIZE)
    if os.path.exists(DATA_FILE):
        print("Loading teacher data...")
        teacher.load_data(DATA_FILE)
    else:
        print("⚠️ Warning: No existing data file found. Please ensure data is added via add_sample() before training.")

    # [修改] 定义分开的 Meter
    ploss_meter = AverageMeter() # Policy Loss
    vloss_meter = AverageMeter() # Value Loss
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    
    step = 0
    print("🚀 Training Started... (Press Ctrl+C to save and exit)")

    try:
        while True:
            # 注意：ReplayBuffer 不再包含 self_play 方法
            # 数据应该通过 add_sample() 方法从外部（如自对弈进程）添加
            
            for _ in range(TRAIN_STEPS):
                step += 1
                try:
                    batch_data = teacher.get_batch(BATCH_SIZE)
                except Exception as e:
                    print(e); break
                
                unique_boards = torch.from_numpy(batch_data[0]).to(device)
                b_idxs_all = torch.from_numpy(batch_data[1]).long().to(device)
                srcs_all   = torch.from_numpy(batch_data[2]).long().to(device)
                dsts_all   = torch.from_numpy(batch_data[3]).long().to(device)
                arrs_all   = torch.from_numpy(batch_data[4]).long().to(device)
                probs_all  = torch.from_numpy(batch_data[5]).float().to(device)
                vs_all     = torch.from_numpy(batch_data[6]).float().to(device)
                
                unique_boards, srcs_all, dsts_all, arrs_all = \
                    apply_consistent_symmetries(unique_boards, srcs_all, dsts_all, arrs_all)

                total_actions = srcs_all.size(0)
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    global_features = nnet_wrapper.nnet.extract_features(unique_boards)
                
                perm = torch.randperm(total_actions, device=device)
                start_idx = 0
                
                # [修改] 分开统计
                total_ce = 0; total_v_loss = 0
                total_weighted_hits = 0; total_weighted_hits_5 = 0; total_prob_sum = 0
                
                nan_found = False 

                while start_idx < total_actions:
                    end_idx = min(start_idx + MAX_ACTIONS_PER_PASS, total_actions)
                    slice_indices = perm[start_idx:end_idx]
                    retain_graph = (end_idx < total_actions)
                    
                    b_idxs_micro = b_idxs_all[slice_indices]
                    srcs_micro   = srcs_all[slice_indices]
                    dsts_micro   = dsts_all[slice_indices]
                    arrs_micro   = arrs_all[slice_indices]
                    probs_micro  = probs_all[slice_indices]
                    vs_micro     = vs_all[b_idxs_micro]
                    
                    with torch.amp.autocast('cuda'):
                        log_p_move, log_p_arrow, out_v = nnet_wrapper.nnet.forward_heads(
                            global_features, srcs_micro, dsts_micro, b_idxs_micro
                        )
                        
                        ce_sum, w_hits, w_hits_5, prob_mass = calculate_metrics(
                            log_p_move, log_p_arrow, srcs_micro, dsts_micro, arrs_micro, probs_micro
                        )
                        
                        # [修正] Value Loss 加权 (Prob Weighted MSE)
                        # 使得复杂局面和残局的 Value 权重平衡
                        raw_mse = (out_v.view(-1) - vs_micro) ** 2
                        weighted_mse = raw_mse * probs_micro
                        v_loss_sum = weighted_mse.sum()
                        
                        loss_micro = (ce_sum + v_loss_sum) / BATCH_SIZE
                    
                    if torch.isnan(loss_micro):
                        print(f"⚠️ NaN detected at Step {step}! Skipping.")
                        nan_found = True
                        break 
                    
                    scaler.scale(loss_micro).backward(retain_graph=retain_graph)
                    
                    total_ce += ce_sum.item()
                    total_v_loss += v_loss_sum.item()
                    total_weighted_hits += w_hits.item()
                    total_weighted_hits_5 += w_hits_5.item()
                    total_prob_sum += prob_mass.item()
                    
                    start_idx = end_idx

                if not nan_found:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(nnet_wrapper.nnet.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    if total_prob_sum < 1e-6: total_prob_sum = 1.0
                    
                    # [修改] 分开更新 Meter
                    # 注意：我们除以 total_actions 来保持与 Loss 计算时的缩放一致
                    ploss_meter.update(total_ce / BATCH_SIZE, BATCH_SIZE)
                    vloss_meter.update(total_v_loss / BATCH_SIZE, BATCH_SIZE)
                    acc1_meter.update(total_weighted_hits / total_prob_sum, total_actions)
                    acc5_meter.update(total_weighted_hits_5 / total_prob_sum, total_actions)
                else:
                    optimizer.zero_grad()
                
                if step % 50 == 0:
                    # [修改] 打印 L_pi 和 L_v
                    print(f"Step {step} | LR: {constant_lr:.6f} | "
                          f"L_pi: {ploss_meter.avg:.4f} L_v: {vloss_meter.avg:.4f} | "
                          f"Acc1: {acc1_meter.avg*100:.2f}% Acc5: {acc5_meter.avg*100:.2f}%")
                    # Reset all meters
                    ploss_meter.reset(); vloss_meter.reset(); acc1_meter.reset(); acc5_meter.reset()

            # Periodic Save
            teacher.save_data(DATA_FILE)
            torch.save({'state_dict': nnet_wrapper.nnet.state_dict()}, os.path.join(CHECKPOINT_DIR, "best_joint.pth.tar"))

    except KeyboardInterrupt:
        print("\n\n🛑 Ctrl+C detected! Saving model state before exit...")
        
        try:
            print("💾 Saving Teacher Data...")
            teacher.save_data(DATA_FILE)
        except Exception as e:
            print(f"Failed to save teacher data: {e}")

        try:
            print("💾 Saving Neural Network...")
            save_path_interrupted = os.path.join(CHECKPOINT_DIR, "interrupted.pth.tar")
            save_path_best = os.path.join(CHECKPOINT_DIR, "best_joint.pth.tar")
            
            state = {'state_dict': nnet_wrapper.nnet.state_dict()}
            torch.save(state, save_path_interrupted)
            torch.save(state, save_path_best)
            print(f"✅ Models saved to:\n  - {save_path_interrupted}\n  - {save_path_best}")
        except Exception as e:
            print(f"Failed to save network: {e}")

        print("👋 Exiting gracefully.")
        sys.exit(0)

if __name__ == "__main__":
    main()