#!/usr/bin/env python3

"""
循环训练控制器 (ResNet-40 适配版 - 修复参数优先级)
"""

import os
import sys
import time
import subprocess
import signal
import psutil
import argparse
import glob
from datetime import datetime

def kill_all_python_processes_except_current():
    """杀掉除了当前进程以外的所有Python进程 (清理显存和僵尸进程)"""
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                if proc.info['pid'] != current_pid and len(proc.info['cmdline']) > 0:
                    # 匹配常见的训练脚本关键词
                    if any('python' in cmd.lower() for cmd in proc.info['cmdline']) or \
                       any('starttrain' in cmd.lower() for cmd in proc.info['cmdline']) or \
                       any('train' in cmd.lower() for cmd in proc.info['cmdline']):
                        # print(f"终止进程: PID {proc.info['pid']}")
                        proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    # 等待进程彻底释放资源
    time.sleep(2)

def run_training_cycle(args):
    """运行一个完整的训练循环"""
    print("\n" + "="*60)
    print(f"启动子进程: train.py (PID: {os.getpid()})")
    print("="*60)

    checkpoint_dir = "./checkpoint"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 基础命令
    train_cmd = [
        sys.executable,
        "train.py",
        "--n-to-1",
        "--load-best" 
    ]

    # --- 参数透传逻辑 ---
    # 只有当 args.xxx 不为 None 时，才添加到命令中
    # 这样 train.py 就会使用 train_config.py 中的配置
    
    if args.num_iters is not None:
        train_cmd.extend(["--num-iters", str(args.num_iters)])
        
    if args.num_eps is not None:
        train_cmd.extend(["--num-eps", str(args.num_eps)])
        
    if args.num_mcts_sims is not None:
        train_cmd.extend(["--num-mcts-sims", str(args.num_mcts_sims)])
    
    # [网络超参数]
    if args.num_channels is not None:
        train_cmd.extend(["--num-channels", str(args.num_channels)])
        
    if args.num_res_blocks is not None:
        train_cmd.extend(["--num-res-blocks", str(args.num_res_blocks)])
        
    if args.learning_rate is not None:
        train_cmd.extend(["--learning-rate", str(args.learning_rate)])
        
    if args.batch_size is not None:
        train_cmd.extend(["--batch-size", str(args.batch_size)])
    
    if args.num_actors is not None:
        train_cmd.extend(["--num-actors", str(args.num_actors)])

    print(f"执行命令: {' '.join(train_cmd)}")

    try:
        # 启动训练进程
        # cwd 设置为当前脚本所在目录 (通常是 amazons/ 父目录或 amazons/ 目录，视你的结构而定)
        # 这里假设 starttrain.py 和 train.py 在同一级 amazons/ 目录下
        process = subprocess.Popen(train_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        # 等待子进程结束
        process.wait()

        if process.returncode != 0:
            print(f"训练循环异常退出, Code: {process.returncode}")
            return False

        print("训练循环成功结束")
    except Exception as e:
        print(f"执行出错: {e}")
        return False

    return True

def main():
    parser = argparse.ArgumentParser(description='亚马逊棋循环训练控制器')
    
    # [关键修改] 将大部分默认值设为 None
    # 这样如果不填，就不会覆盖 train_config.py 里的设置
    
    # num-iters 默认为 1 是合理的，因为 train.py 内部循环通常只跑 1 次，外部由 starttrain 控制循环
    parser.add_argument('--num-iters', type=int, default=1, help='单次运行迭代次数')
    
    parser.add_argument('--num-eps', type=int, default=None, help='每轮对局数 (默认: 读取config)')
    parser.add_argument('--num-mcts-sims', type=int, default=None, help='MCTS模拟次数 (默认: 读取config)')
    
    parser.add_argument('--num-channels', type=int, default=None, help='ResNet通道数 (默认: 读取config)')
    parser.add_argument('--num-res-blocks', type=int, default=None, help='ResNet残差块数 (默认: 读取config)')
    parser.add_argument('--batch-size', type=int, default=None, help='训练批次大小 (默认: 读取config)')
    parser.add_argument('--learning-rate', type=float, default=None, help='学习率 (默认: 读取config)')
    
    parser.add_argument('--num-actors', type=int, default=None, help='并行演员数 (默认: 读取config)')
    
    # 控制器本身的参数，需要默认值
    parser.add_argument('--num-cycles', type=int, default=1000, help='总循环轮数')
    parser.add_argument('--clean-processes', action='store_true', help='每次循环前清理僵尸进程')

    args = parser.parse_args()
    
    print(">>> 循环训练控制器启动 <<<")
    print("提示: 未指定的参数将直接读取 train_config.py 中的配置")
    
    cycle_count = 0
    while True:
        cycle_count += 1
        print(f"\n>>> Cycle {cycle_count}/{args.num_cycles} <<<")
        
        if args.clean_processes:
            kill_all_python_processes_except_current()
        
        success = run_training_cycle(args)
        
        if not success:
            print("本轮训练失败，尝试重试或退出...")
            time.sleep(5)
            # 如果遇到连续失败，可以考虑 break
        
        if cycle_count >= args.num_cycles:
            break
        
        # 短暂休息让 GPU 降温/释放显存
        time.sleep(2)

if __name__ == "__main__":
    main()