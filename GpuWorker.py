import torch
import torch.nn.functional as F
import numpy as np
import multiprocessing as mp
import queue
import time
import logging
import traceback 
import sys
import os
import psutil

log = logging.getLogger(__name__)

# =============================================================================
# [核心] C++ 模块加载与日志检测
# =============================================================================
print("\n" + "="*60)
print("[GpuWorker] 正在初始化特征提取模块...")

try:
    import amazons_ops
    CPP_FEATURE_AVAILABLE = True
    
    # 获取模块文件路径（确认加载的是哪一个文件）
    module_path = getattr(amazons_ops, '__file__', 'Unknown location')
    
    print(f"[GpuWorker] ✅ C++ Extension 加载成功!")
    print(f"[GpuWorker] 📍 模块路径: {module_path}")
    print("[GpuWorker] 🚀 模式: 高性能 C++ 特征提取 (与蒸馏训练保持 100% 一致)")

except ImportError as e:
    CPP_FEATURE_AVAILABLE = False
    print(f"[GpuWorker] ❌ C++ Extension 加载失败!")
    print(f"[GpuWorker] ⚠️ 错误信息: {e}")
    print("[GpuWorker] 🐢 模式: Python 慢速兼容模式 (警告：如逻辑不一致会导致概率异常)")
    
print("="*60 + "\n")

def set_high_priority():
    try:
        p = psutil.Process(os.getpid())
        if os.name == 'nt': p.nice(psutil.HIGH_PRIORITY_CLASS)
        else: p.nice(-10)
    except Exception: pass

# =============================================================================
# Python 回退逻辑 (仅当 C++ 加载失败时使用)
# =============================================================================
def get_mobility_batch_python(pcs, obs):
    B, H, W = pcs.shape
    mob = np.zeros((B, H, W), dtype=np.float32)
    dirs = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    for i in range(B):
        p_locs = np.argwhere(pcs[i] > 0)
        for r, c in p_locs:
            for dr, dc in dirs:
                cr, cc = r+dr, c+dc
                while 0 <= cr < 8 and 0 <= cc < 8:
                    if obs[i, cr, cc] != 0: break
                    mob[i, cr, cc] = 1.0
                    cr += dr
                    cc += dc
    return mob

def get_7ch_features_batch_python(board_3ch):
    B = board_3ch.shape[0]
    my_map = board_3ch[:, 0]
    op_map = board_3ch[:, 1]
    obstacles = (my_map + op_map + board_3ch[:, 2]) > 0.5
    obstacles = obstacles.astype(np.float32)

    my_mob = get_mobility_batch_python(my_map, obstacles)
    op_mob = get_mobility_batch_python(op_map, obstacles)
    my_pot = get_mobility_batch_python(my_mob, obstacles)
    op_pot = get_mobility_batch_python(op_mob, obstacles)

    # 堆叠顺序必须与 C++ write_board_to_numpy 保持一致
    res = np.stack([my_map, op_map, obstacles, my_mob, op_mob, my_pot, op_pot], axis=1)
    return res

# =============================================================================
# 混合特征提取器
# =============================================================================

def get_7ch_features_batch(board_3ch):
    """
    智能选择 C++ 或 Python 进行特征计算
    """
    if CPP_FEATURE_AVAILABLE:
        try:
            B = board_3ch.shape[0]
            board_7ch = np.zeros((B, 7, 8, 8), dtype=np.float32)
            
            # C++ 接口目前只支持单板输入，循环调用 C++ 函数
            # 由于 C++ 内部位运算极快，这通常比 Python 批处理还要快
            for i in range(B):
                my = board_3ch[i, 0].astype(np.int32)
                op = board_3ch[i, 1].astype(np.int32)
                arr = board_3ch[i, 2].astype(np.int32)
                
                board_7ch[i] = amazons_ops.compute_7ch_features(my, op, arr)
            
            return board_7ch
        except Exception as e:
            print(f"[GpuWorker] ⚠️ C++ Execution Error: {e}. Falling back to Python once.")
            return get_7ch_features_batch_python(board_3ch)
    else:
        return get_7ch_features_batch_python(board_3ch)

def encode_batch_one_hot(boards_numpy):
    """Step 1: 基础 3 通道编码 (My, Op, Arrow)"""
    if not isinstance(boards_numpy, np.ndarray):
        boards_numpy = np.array(boards_numpy)
    layer_my = (boards_numpy == 1).astype(np.float32)
    layer_op = (boards_numpy == -1).astype(np.float32)
    # 0=Empty, 1=My, -1=Op. 任何非这三个值的都被视为 Arrow (通常是 2)
    layer_arr = (~np.isin(boards_numpy, [0, 1, -1])).astype(np.float32)
    return np.stack([layer_my, layer_op, layer_arr], axis=1)

# =============================================================================
# Worker Class
# =============================================================================

class GpuWorker:
    def __init__(self, game_class_name, game_params, network_class_name, network_params, args_dict):
        if game_class_name == "AmazonsGame":
            from amazons.AmazonsGame import AmazonsGame as GameClass
        else: raise ValueError(f"Unknown game class: {game_class_name}")

        if network_class_name == "NNetWrapper":
            from amazons.pytorch.NNet import NNetWrapper as NetworkClass
        else: raise ValueError(f"Unknown network class: {network_class_name}")

        class ArgsObj:
            def __init__(self, d):
                for k, v in d.items(): setattr(self, k, v)

        args = ArgsObj(args_dict)
        self.game = GameClass(**game_params)
        self.nnet = NetworkClass(self.game, args)

        if 'model_path' in network_params:
            try:
                full_path = os.path.join(network_params['checkpoint'], network_params['filename'])
                print(f"[GpuWorker] 📥 正在加载模型权重: {full_path}")
                
                if not os.path.exists(full_path):
                    print(f"[GpuWorker] ❌ 模型文件不存在! GpuWorker 将退出。请检查 checkpoint 路径。")
                    sys.exit(1)
                    
                self.nnet.load_checkpoint(network_params['checkpoint'], network_params['filename'])
                print(f"[GpuWorker] ✅ 模型权重加载成功。")
            except Exception as e:
                log.error(f"GPU Worker load model failed: {e}")
                sys.exit(1)

        self.nnet.nnet.eval()
        self.nnet.nnet.to(self.nnet.device)
        print(f"[GpuWorker] Started on Device: {self.nnet.device}")

    def run(self, gpu_work_queue, gpu_result_queues_list):
        set_high_priority()
        
        try:
            while True:
                try:
                    # 1. 获取任务
                    boards_numpy, dispatcher_id = gpu_work_queue.get(timeout=2.0)

                    # 2. 预处理: 3ch -> 7ch (C++)
                    boards_3ch = encode_batch_one_hot(boards_numpy)
                    boards_7ch = get_7ch_features_batch(boards_3ch)
                    
                    boards_tensor = torch.from_numpy(boards_7ch).float().to(self.nnet.device)

                    # 3. 推理 (Model B)
                    with torch.no_grad():
                        # Model B forward returns: log_p_move, log_p_arrow, v
                        # log_p_move: (B, 4096) - LogSoftmaxed
                        # log_p_arrow: (B, 4096, 64) - LogSoftmaxed
                        # v: (B, 1)
                        
                        log_p_move, log_p_arrow, v = self.nnet.nnet(boards_tensor)
                        
                        # 转为概率 (Exp) 供 MCTS 使用
                        # 注意：p_arrow 矩阵很大 (B * 4096 * 64)，传输可能会成为瓶颈
                        # 如果 IPC 卡顿，后续可在 MCTS 端接收 Logits，或者在这里只传 Top-K
                        p_move = torch.exp(log_p_move).data.cpu().numpy()
                        p_arrow = torch.exp(log_p_arrow).data.cpu().numpy()
                        vs = v.data.cpu().numpy()
                        
                    # 4. 发送结果
                    if 0 <= dispatcher_id < len(gpu_result_queues_list):
                        # 结构: ((p_move, p_arrow), vs)
                        gpu_result_queues_list[dispatcher_id].put(((p_move, p_arrow), vs.astype(np.float64)))
                    
                except queue.Empty:
                    continue
        
        except Exception as e:
            print(f"\n!!! GPU WORKER CRASHED !!! Error: {e}", flush=True)
            traceback.print_exc()
            sys.exit(1) 

def gpu_worker_process_main(gpu_work_queue, gpu_result_queues_list, args_dict, game_class_name, game_params, network_class_name, network_params):
    worker = GpuWorker(game_class_name, game_params, network_class_name, network_params, args_dict)
    worker.run(gpu_work_queue, gpu_result_queues_list)

def start_gpu_worker_process(gpu_work_queue, gpu_result_queues_list, args_obj, game_class_name, game_params, network_class_name, network_params):
    args_dict = {}
    for attr in dir(args_obj):
        if not attr.startswith('_') and not callable(getattr(args_obj, attr)):
            try:
                val = getattr(args_obj, attr)
                if isinstance(val, (str, int, float, bool, list, dict, tuple)) or val is None:
                    args_dict[attr] = val
            except: pass

    gpu_worker_proc = mp.Process(
        target=gpu_worker_process_main,
        args=(gpu_work_queue, gpu_result_queues_list, args_dict, game_class_name, game_params, network_class_name, network_params)
    )
    gpu_worker_proc.start()
    return gpu_worker_proc