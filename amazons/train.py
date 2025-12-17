#!/usr/bin/env python3
"""
亚马逊棋 AI 训练主程序 (Model B / C++ Buffer 适配版)
"""

import os
import sys
import argparse
import logging
import multiprocessing
import time

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amazons.AmazonsGame import AmazonsGame
from amazons.train_config import NTo1TrainingConfig, FastTrainingConfig, LongTermTrainingConfig
from OrchestratedParallelCoach import OrchestratedParallelCoach
from amazons.pytorch.NNet import NNetWrapper

def setup_logging():
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    if not os.path.exists(logs_dir): os.makedirs(logs_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f'training_{timestamp}.log')
        
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        logger.info(f"Log initialized: {log_file}")

    return logger
import numpy as np
import torch
def debug_model_health(nnet, game, logger):
    """
    模型加载后的“体检”函数：检查权重状态和Value Head的逻辑直觉
    [修复版] 增加了 float() 转换以防止 numpy 格式化报错
    """
    logger.info("\n" + "="*60)
    logger.info(">>> 🩺 正在执行模型健全性检查 (Sanity Check) <<<")
    
    # --- 1. 静态权重检查 ---
    try:
        total_params = 0
        non_zero = 0
        has_nan = False
        
        for name, param in nnet.nnet.named_parameters():
            if param.requires_grad:
                n = param.numel()
                total_params += n
                non_zero += torch.count_nonzero(param).item()
                if torch.isnan(param).any():
                    has_nan = True
        
        zero_ratio = 1.0 - (non_zero / total_params)
        logger.info(f"[1] 权重统计:")
        logger.info(f"    - 参数总量: {total_params}")
        logger.info(f"    - 零值比例: {zero_ratio:.2%} (若接近100%说明模型可能未初始化)")
        
        if has_nan:
            logger.error("    - 状态: ❌ 失败 (包含 NaN)")
        elif zero_ratio > 0.99:
            logger.warning("    - 状态: ⚠️ 警告 (权重几乎全为0)")
        else:
            logger.info("    - 状态: ✅ 正常 (数值分布合理)")
            
    except Exception as e:
        logger.error(f"权重检查出错: {e}")

    # --- 2. 动态逻辑检查 (Value Head) ---
    logger.info(f"[2] 局面评估测试 (Value Head范围: -1.0 ~ 1.0):")
    
    try:
        # A. 初始局面（从 Player 1 视角）
        board_init = game.getInitBoard()
        board_init_can = game.getCanonicalForm(board_init, 1)
        _, v_init = nnet.predict(board_init_can)
        
        # B. 白棋(Player 1)绝对优势（从 Player 1 视角）
        board_p1_win = np.copy(board_init)
        board_p1_win[board_p1_win == -1] = 0 
        board_p1_win_can = game.getCanonicalForm(board_p1_win, 1)
        _, v_p1 = nnet.predict(board_p1_win_can)
        
        # C. 黑棋(Player -1)绝对优势
        board_p2_win = np.copy(board_init)
        board_p2_win[board_p2_win == 1] = 0 
        board_p2_win_can = game.getCanonicalForm(board_p2_win, -1)
        _, v_p2_raw = nnet.predict(board_p2_win_can)
        # v_p2_raw 是“当前玩家=黑”的视角，要转换为“站在白棋视角”的值
        v_p2 = -v_p2_raw

        # [关键修改] 使用 float() 将 numpy array 转为 python float
        v_init = float(v_init)
        v_p1 = float(v_p1)
        v_p2 = float(v_p2)

        logger.info(f"    - 初始局面 v值: {v_init:.4f} \t[预期: 接近 0.0]")
        logger.info(f"    - 白棋碾压 v值: {v_p1:.4f} \t[预期: 接近 1.0 (白胜)]")
        logger.info(f"    - 黑棋碾压 v值: {v_p2:.4f} \t[预期: 接近 -1.0 (黑胜)]")

        if v_p1 > 0.5 and v_p2 < -0.5:
            logger.info("    - 逻辑判定: ✅ 通过 (模型能区分优劣势)")
        else:
            logger.warning("    - 逻辑判定: ⚠️ 存疑 (模型区分度不足，或处于训练早期)")

    except Exception as e:
        logger.error(f"推理测试出错: {e}")
        import traceback
        traceback.print_exc()
        
    logger.info("="*60 + "\n")

def main():
    logger = setup_logging()
    logger.info(">>> 启动亚马逊棋 AlphaZero 训练系统 (Model B Deep Fusion) <<<")

    parser = argparse.ArgumentParser(description='亚马逊棋 AI 训练')
    
    # --- 模式选择 ---
    parser.add_argument('--fast', action='store_true', help='快速调试模式')
    parser.add_argument('--long', action='store_true', help='长期训练模式')
    parser.add_argument('--n-to-1', action='store_true', help='N-to-1 架构 (默认)')
    
    # --- 流程控制 ---
    parser.add_argument('--load', action='store_true', help='加载最新的检查点')
    parser.add_argument('--load-best', action='store_true', help='加载最佳模型')
    parser.add_argument('--selfplay-only', action='store_true', help='只进行自对弈')
    parser.add_argument('--training-only', action='store_true', help='只进行训练')
    parser.add_argument('--arena-only', action='store_true', help='只进行竞技场对比')
    
    # --- 核心超参数覆盖 ---
    parser.add_argument('--num-iters', type=int, help='迭代次数')
    parser.add_argument('--num-eps', type=int, help='自对弈局数')
    parser.add_argument('--num-actors', type=int, help='并行 CPU 进程数')
    parser.add_argument('--num-mcts-sims', type=int, help='MCTS 模拟次数')
    parser.add_argument('--num-channels', type=int, default=256, help='网络通道数')
    parser.add_argument('--num-res-blocks', type=int, default=20, help='残差块数量')
    parser.add_argument('--batch-size', type=int, default=256, help='训练 Batch Size') # [修正] 默认256
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='学习率')     # [修正] 默认2e-4
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout 概率')

    args = parser.parse_args()

    # 1. 初始化游戏
    game = AmazonsGame(8)

    # 2. 加载配置
    if args.fast:
        logger.info("配置: FastTrainingConfig (调试)")
        config = FastTrainingConfig()
    elif args.long:
        logger.info("配置: LongTermTrainingConfig (长期)")
        config = LongTermTrainingConfig()
    else:
        logger.info("配置: NTo1TrainingConfig (标准)")
        config = NTo1TrainingConfig()

    # 3. 应用参数覆盖
    if args.num_iters: config.numIters = args.num_iters
    if args.num_eps: config.numEps = args.num_eps
    if args.num_actors: config.num_actors = args.num_actors
    if args.num_mcts_sims: config.numMCTSSims = args.num_mcts_sims
    if args.num_channels: config.num_channels = args.num_channels
    if args.num_res_blocks: config.num_res_blocks = args.num_res_blocks
    if args.batch_size: config.training_batch_size = args.batch_size
    if args.learning_rate: config.learningRate = args.learning_rate
    if args.dropout: config.dropout = args.dropout

    logger.info(f"网络架构: ResNet-{getattr(config, 'num_res_blocks', 20)*2} ({getattr(config, 'num_channels', 256)} ch)")
    logger.info(f"训练批次: {getattr(config, 'training_batch_size', 256)}")

    # 4. 设置多进程启动方式
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 5. 初始化神经网络
    nnet = NNetWrapper(game, config)

    if args.load_best:
        config.load_model = True
        config.load_folder_file = (config.checkpoint, 'best.pth.tar')
    elif args.load:
        config.load_model = True
        
    if config.load_model:
        logger.info(f"尝试加载模型: {config.load_folder_file}")
        try:
            if not os.path.exists(os.path.join(config.load_folder_file[0], config.load_folder_file[1])):
                 logger.warning("指定模型不存在，将从头开始")
            nnet.load_checkpoint(config.load_folder_file[0], config.load_folder_file[1])
        except Exception as e:
            logger.error(f"主进程模型加载提示 (可忽略): {e}")

    # 6. 初始化 Coach
    coach = OrchestratedParallelCoach(game, nnet, config)
    debug_model_health(nnet, game, logger)
    # 7. 执行流程
    if args.arena_only:
        coach.run_arena()
        return

    if args.training_only:
        # 直接调用 Training，假设数据已经在 C++ Buffer 里了 (teacher_data_joint.bin)
        logger.info(">>> 仅训练模式 (Training Only) <<<")
        coach.run_training()
        return

    # 默认流程: Self-Play -> Training -> Arena
    logger.info(">>> 启动完整 Self-Play 循环 <<<")
    coach.run_selfplay()

    logger.info("所有任务完成。")

if __name__ == "__main__":
    main()