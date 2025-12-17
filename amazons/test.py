import torch
import numpy as np
import sys
import os

# ---------------------------------------------------------
# 1. 路径修正 (Path Correction)
# ---------------------------------------------------------
current_dir = os.getcwd()
print(f"[INFO] Current Working Directory: {current_dir}")

# 添加当前目录到 path (确保能找到 amazons_ops 和 train_config)
sys.path.append(current_dir)

# 添加 pytorch 子目录到 path (确保能找到 AmazonsPytorch)
pytorch_dir = os.path.join(current_dir, 'pytorch')
sys.path.append(pytorch_dir)

# ---------------------------------------------------------
# 2. 引入模块
# ---------------------------------------------------------
try:
    # 从子目录 pytorch/AmazonsPytorch.py 加载
    from AmazonsPytorch import AmazonsNNet
    print("[SUCCESS] AmazonsNNet imported from pytorch subdir.")
except ImportError as e:
    print(f"[ERROR] Cannot import AmazonsNNet: {e}")
    print(f"       Checked path: {pytorch_dir}")
    sys.exit(1)

try:
    # 尝试从当前目录导入 train_config
    from train_config import NTo1TrainingConfig
    print("[SUCCESS] NTo1TrainingConfig imported.")
except ImportError:
    try:
        # 回退尝试：如果 train_config 在上一级目录
        sys.path.append(os.path.dirname(current_dir))
        from amazons.train_config import NTo1TrainingConfig
        print("[SUCCESS] NTo1TrainingConfig imported (from package).")
    except ImportError as e:
        print(f"[ERROR] Cannot import train_config: {e}")
        sys.exit(1)

try:
    import amazons_ops
    print("[SUCCESS] C++ amazons_ops imported.")
except ImportError:
    print("[ERROR] amazons_ops NOT found. Please compile the C++ extension first using 'python setup.py build_ext --inplace'")
    sys.exit(1)

# ---------------------------------------------------------
# 3. 验证逻辑主体
# ---------------------------------------------------------
def verify_model_behavior():
    print("\n" + "="*60)
    print(">>> 神经网络推理验证 (Model Inference Check) <<<")
    print(">>> 场景：黑棋被困死 (Op Dead) vs 白棋大优势 (My Free)")
    print("="*60)

    # --- 1. 准备数据 (Extreme Asymmetric Trap) ---
    board_my  = np.zeros((8, 8), dtype=np.int32) 
    board_op  = np.zeros((8, 8), dtype=np.int32) 
    board_arr = np.zeros((8, 8), dtype=np.int32) 

    # 黑棋被困在上半场
    black_starts = [(0, 2), (0, 5), (2, 0), (2, 7)]
    for r, c in black_starts: board_op[r, c] = 1
    # 填满上半场障碍 (除了黑棋自己)
    for r in range(4):
        for c in range(8):
            if board_op[r, c] == 0: board_arr[r, c] = 1

    # 白棋在下半场自由
    white_starts = [(7, 2), (7, 5), (5, 0), (5, 7)]
    for r, c in white_starts: board_my[r, c] = 1

    # C++ 提取特征
    print("[INFO] Computing C++ Features...")
    features_np = amazons_ops.compute_7ch_features(board_my, board_op, board_arr)
    
    # 转为 Tensor (Batch Size = 1)
    input_tensor = torch.from_numpy(features_np).unsqueeze(0).float()
    
    # --- 2. 加载模型 ---
    config = NTo1TrainingConfig()
    config.num_channels = 256 
    config.num_res_blocks = 20 
    
    # 模拟一个伪 Game 对象获取尺寸
    class StubGame:
        def getBoardSize(self): return (8, 8)
    
    print(f"[INFO] Initializing ResNet-{config.num_res_blocks*2}...")
    nnet = AmazonsNNet(StubGame(), config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using Device: {device}")
    nnet.to(device)
    input_tensor = input_tensor.to(device)
    
    # 寻找 Checkpoint
    # 既然你在 amazons 目录下，checkpoint 可能在 amazons/checkpoint 或 ../checkpoint
    # 这里假设是 amazons/checkpoint
    checkpoint_dir = os.path.join(current_dir, "checkpoint")
    checkpoint_path = os.path.join(checkpoint_dir, "best.pth.tar")
    
    MODEL_TRAINED = False
    if os.path.exists(checkpoint_path):
        print(f"[INFO] Loading Checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            nnet.load_state_dict(checkpoint['state_dict'])
            MODEL_TRAINED = True
            print("[SUCCESS] Model loaded.")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
    else:
        print(f"[WARN] Checkpoint not found at {checkpoint_path}")
        print("       Running with random weights for shape verification.")

    nnet.eval()

    # --- 3. 执行推理 ---
    print("[INFO] Executing Forward Pass...")
    with torch.no_grad():
        # inference mode
        log_p_move, log_p_arrow, v = nnet(input_tensor)

    # --- 4. 验证形状 ---
    print("\n[Check 1] Shape Verification")
    print(f"   LogP Move : {log_p_move.shape} (Expect: [1, 4096] or [1, 64, 64])")
    print(f"   LogP Arrow: {log_p_arrow.shape} (Expect: [1, 4096, 64] or similar)")
    print(f"   Value     : {v.shape} (Expect: [1, 1])")

    # --- 5. 验证 Value ---
    print("\n[Check 2] Value Head Logic")
    value_scalar = v.item()
    print(f"   Predicted Value: {value_scalar:.4f} (Range: -1.0 to 1.0)")
    print("   Scene: White (My) is Free, Black (Op) is Trapped.")
    
    if not MODEL_TRAINED:
        print("   (Random Weights - Value meaningless)")
    else:
        if value_scalar > 0.5:
            print("✅ PASS: Model predicts advantage for Current Player (White).")
        elif value_scalar < -0.5:
            print("❌ FAIL: Model predicts LOSS for Current Player. Sign might be flipped!")
        else:
            print("⚠️ INDETERMINATE: Value is neutral. Training might be insufficient.")

    # --- 6. 验证 Policy ---
    print("\n[Check 3] Policy Head Sanity")
    # Move Prob
    if log_p_move.dim() == 2: # (B, 4096)
        move_probs = torch.exp(log_p_move).cpu().numpy().flatten()
    else: 
        move_probs = torch.exp(log_p_move).cpu().numpy().flatten()
        
    best_move_idx = np.argmax(move_probs)
    src = best_move_idx // 64
    dst = best_move_idx % 64
    r_src, c_src = src // 8, src % 8
    
    print(f"   Best Move Source: ({r_src}, {c_src}) -> Dst: {dst}")
    if board_my[r_src, c_src] == 1:
        print(f"   Valid Source? ✅ YES (It is My Piece)")
    else:
        print(f"   Valid Source? ❌ NO (Trying to move empty/enemy/obstacle)")

if __name__ == "__main__":
    verify_model_behavior()