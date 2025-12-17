<div align="center">

Amazons Chess AlphaZero Implementation

亚马逊棋 AlphaZero 高性能实现

<p align="center">
<strong>Python (PyTorch) + C++ 混合架构 | 注意力机制 (Attention) | 动态特征更新 (Dynamic Feature Update)</strong>
</p>

</div>

📖 项目简介 (Introduction)

本项目是一个基于 AlphaZero 范式的亚马逊棋（Game of Amazons）高性能 AI 实现。

针对亚马逊棋巨大的动作空间分支因子问题，本项目采用 Python (PyTorch) + C++ 混合架构，通过引入 注意力机制（Attention Mechanism） 和 动态特征更新（Dynamic Feature Update） 网络架构，成功实现了完全通过自我对弈（Self-Play）进化的强化学习系统。

📚 技术演进与开发日志 (Development Blog)

本项目记录了从零构建高性能 AlphaZero 的完整心路历程。以下技术博客按时间线梳理了架构的迭代与进化：

日期

标题

简介

2025-11-17

Amazon棋强化学习模型AlphaZero训练

探索 AlphaZero 算法在亚马逊棋中的初步落地，建立基础的训练管线与环境搭建。

2025-11-20

凤凰涅槃：AlphaZero结构重铸

针对早期版本的性能瓶颈进行底层重构，优化 MCTS 并行策略与数据流转效率。

2025-11-23

引入注意力机制，亚马逊棋Bot神经网络架构再颠覆

[核心技术] 详解如何设计 Dynamic Feature Update Head，利用 Attention 解决“移动+射箭”联合动作空间的稀疏性难题。

2025-11-24

互动网页：全面理解我是如何构建属于亚马逊棋的AlphaZero的

通过直观的交互式演示，解构模型的决策逻辑与构建过程。

✨ 核心特性 (Key Features)

🚀 高性能 C++ 扩展 底层游戏逻辑 (GameCore) 和特征提取 (FeatureExtractor) 采用 C++ 位运算（Bitboard）实现，大幅提升 MCTS 模拟速度。

🧠 动态特征更新网络 摒弃传统的 Flatten 输出，采用 ResNet + Attention 架构，在隐空间中动态模拟“移动后射箭”的价值流转。

🔄 并行自对弈 支持多进程并行 MCTS 搜索，配合 GPU 批量推理，最大化硬件利用率。

💾 高效经验回放 C++ 实现的线程安全 Replay Buffer，支持大规模训练数据的无锁存取。

📂 项目结构 (Project Structure)
```

.
├── amazons/                       # Python 核心逻辑
│   ├── AmazonsGame.py             # 游戏规则封装
│   ├── pytorch/                   # 神经网络架构
│   │   ├── AmazonsPytorch.py      # ResNet + Dynamic Head 定义
│   │   └── NNet.py                # 网络训练 Wrapper
│   └── train_config.py            # 训练超参数配置
├── amazons_ops.cpp                # C++ 核心扩展 (特征提取、Buffer、规则引擎)
├── OrchestratedMCTS.py            # 并行 MCTS 搜索逻辑
├── OrchestratedParallelCoach.py   # 训练调度器 (Self-Play & Arena)
├── botzone.py                     # 在线评测提交脚本 (Botzone适配)
└── setup.py                       # 编译配置文件

```
🛠️ 安装与编译 (Installation)

环境要求

Python >= 3.8

PyTorch >= 1.9.0 (CUDA 推荐)

C++ 编译器 (GCC/Clang/MSVC) 支持 C++17

1. 克隆仓库
```bash
git clone [<your-repo-url>](https://github.com/ly-geming/amazons-chess-Attentionalphazero)
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 编译 C++ 高性能扩展 (重要)

这是项目运行的核心，必须首先完成。
```bash
python setup.py build_ext --inplace

```
编译成功后，目录中应出现 amazons_ops.pyd (Windows) 或 amazons_ops.so (Linux)。

🚀 快速开始 (Quick Start)

启动 AlphaZero 自我博弈训练

该命令将启动完整的强化学习循环：自对弈收集数据 -> 神经网络训练 -> 竞技场评估 -> 模型迭代。

# 推荐使用 load-best 继续优化最佳模型
```bash
python amazons/train.py --load-best
```
# 或者从零开始训练
```bash
python amazons/train.py
```

模型验证

运行推理验证脚本，检查模型是否理解复杂的“困毙”规则以及 Attention 机制是否正常工作。
```bash
python verify_model_inference.py
```

⚙️ 核心组件开发 (Development)

调用 C++ 特征提取
```python
import amazons_ops
import numpy as np
```
# 返回 (7, 8, 8) 的浮点特征张量，包含：
# MyPiece, OpPiece, Obstacles, MyMobility, OpMobility, MyShoot, OpShoot
```python
features = amazons_ops.compute_7ch_features(board_my, board_op, board_arr)
```

调用 Replay Buffer
```python
buffer = amazons_ops.ReplayBuffer(capacity=1000000)
```
# 线程安全地添加自对弈样本
```python
buffer.add_sample(board, player, winner, srcs, dsts, arrs, probs)
```

🙏 致谢 (Acknowledgements)

特别致谢：诚挚感谢我的 《计算概论A》 课程的助教。本次大作业不仅为我提供了深入探索神经网络与 AlphaZero 算法的宝贵契机，助教们的悉心指导更在项目实现过程中给予了我巨大的帮助与启发。

算法灵感来源于 DeepMind 的 AlphaZero 论文：Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm

感谢开源社区对 AlphaZero General 框架的探索，本项目在此基础上针对亚马逊棋进行了深度定制。

📜 许可证 (License)

本项目采用 MIT License 许可证。
