import os
import sys
import pickle
import numpy as np
from tqdm import tqdm

# 确保能找到 amazons_ops
sys.path.append(os.getcwd())

try:
    import amazons_ops
    print("[SUCCESS] C++ amazons_ops module loaded.")
except ImportError:
    print("[ERROR] Could not load amazons_ops. Please compile it first.")
    sys.exit(1)

def convert_bin_to_examples(bin_path, output_dir, samples_per_file=50000):
    """
    将 C++ 的 .bin 训练数据转换为 AlphaZero 的 .examples 文件
    关键逻辑：将 C++ 的 7通道特征 还原为 8x8 原始棋盘，以适配 NNet.py 的输入要求
    """
    if not os.path.exists(bin_path):
        print(f"Error: Binary file not found at {bin_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading binary data from {bin_path}...")
    teacher = amazons_ops.ReplayBuffer(5000000) 
    teacher.load_data(bin_path)
    
    print("Starting conversion...")
    
    # 你的数据很大，我们提取 100万条作为长期记忆
    TOTAL_SAMPLES_TO_EXTRACT = 1000000
    batch_size = 10000
    collected_examples = []
    file_counter = 0
    
    pbar = tqdm(total=TOTAL_SAMPLES_TO_EXTRACT)
    
    while len(collected_examples) < TOTAL_SAMPLES_TO_EXTRACT:
        try:
            # 获取一批数据 (从 C++ 中随机采样)
            # boards shape: (Batch, 7, 8, 8)
            batch_data = teacher.get_batch(batch_size)
            
            boards_7ch = batch_data[0] 
            srcs = batch_data[2]
            dsts = batch_data[3]
            arrs = batch_data[4]
            vs = batch_data[6]
            
            for i in range(batch_size):
                # --- [核心修复] 特征还原逻辑 ---
                # C++ 输出的是 7 通道特征，但 NNet.py 期望输入 8x8 原始棋盘
                # 我们需要根据特征层还原出 My(1), Op(-1), Arrow(2), Empty(0)
                
                feat = boards_7ch[i] # shape (7, 8, 8)
                
                # 初始化空棋盘
                raw_board = np.zeros((8, 8), dtype=np.int32)
                
                # feat[0] 是 My Pieces
                raw_board[feat[0] > 0.5] = 1
                
                # feat[1] 是 Op Pieces
                raw_board[feat[1] > 0.5] = -1
                
                # feat[2] 是 Obstacles (My + Op + Arrow)
                # Arrow = Obstacles - (My + Op)
                # 也就是：在 Obstacles 层有值，但在 My 和 Op 层没值的地方，就是 Arrow
                arrow_mask = (feat[2] > 0.5) & (feat[0] < 0.5) & (feat[1] < 0.5)
                raw_board[arrow_mask] = 2
                
                # --- 构造 Target ---
                target_move = np.zeros((64, 64), dtype=np.float32)
                target_arrow = np.zeros((64, 64), dtype=np.float32)
                
                s = srcs[i]
                d = dsts[i]
                a = arrs[i]
                
                # 这是一个近似：我们只知道最好的一步，所以设为 1.0
                target_move[s][d] = 1.0 
                target_arrow[d][a] = 1.0
                
                v = vs[i]
                
                # --- [保存格式验证] ---
                # 格式必须是: (Board, Pi_Tuple, Value)
                # 其中 Pi_Tuple = (TargetMove, TargetArrow)
                collected_examples.append([raw_board, (target_move, target_arrow), v])
                
                # 分文件保存
                if len(collected_examples) >= samples_per_file:
                    file_counter += 1
                    filename = os.path.join(output_dir, f"distilled_data_{file_counter}.examples")
                    with open(filename, "wb+") as f:
                        pickle.dump(collected_examples, f)
                    collected_examples = [] # 清空缓冲
            
            pbar.update(batch_size)
            
        except Exception as e:
            print(f"Extraction stopped: {e}")
            break
            
    pbar.close()
    
    # 保存剩余数据
    if len(collected_examples) > 0:
        file_counter += 1
        filename = os.path.join(output_dir, f"distilled_data_{file_counter}.examples")
        with open(filename, "wb+") as f:
            pickle.dump(collected_examples, f)
        print(f"Saved final file {filename}")

if __name__ == "__main__":
    # 1. 你的 C++ 数据文件
    BIN_FILE = "teacher_data_joint.bin"  
    
    # 2. 输出目录 (直接设为 checkpoint 方便加载)
    OUTPUT_DIR = "./checkpoint"          
    
    convert_bin_to_examples(BIN_FILE, OUTPUT_DIR)