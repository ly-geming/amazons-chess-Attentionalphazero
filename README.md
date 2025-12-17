<div align="center">

# Amazons Chess AlphaZero Implementation

### 亚马逊棋 AlphaZero 高性能实现

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![C++](https://img.shields.io/badge/C%2B%2B-17-green)
![License](https://img.shields.io/badge/license-MIT-blue)

<p align="center">
<strong>Python (PyTorch) + C++ 混合架构 | 注意力机制 (Attention) | 动态特征更新 (Dynamic Feature Update)</strong>
</p>

</div>

---

## 📖 项目简介 (Introduction)

本项目是一个基于 **AlphaZero** 范式的亚马逊棋（Game of Amazons）高性能 AI 实现。

针对亚马逊棋巨大的动作空间分支因子问题，本项目采用 **Python (PyTorch) + C++ 混合架构**，通过引入 **注意力机制（Attention Mechanism）** 和 **动态特征更新（Dynamic Feature Update）** 网络架构，成功实现了完全通过自我对弈（Self-Play）进化的强化学习系统。

---

## 📚 技术演进与开发日志 (Development Blog)

本项目记录了从零构建高性能 AlphaZero 的完整心路历程。以下技术博客按时间线梳理了架构的迭代与进化：

| 日期 | 标题 | 简介 |
| :--- | :--- | :--- |
| **2025-11-17** | [**Amazon棋强化学习模型AlphaZero训练**](https://ly-geming.github.io/2025/11/17/Amazon%E6%A3%8B%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83/) | 探索 AlphaZero 算法在亚马逊棋中的初步落地，建立基础的训练管线与环境搭建。 |
| **2025-11-20** | [**凤凰涅槃：AlphaZero结构重铸**](https://ly-geming.github.io/2025/11/20/%E5%87%A4%E5%87%B0%E6%B6%85%E6%A7%83-AlphaZero%E7%BB%93%E6%9E%84%E9%87%8D%E9%93%B8/) | 针对早期版本的性能瓶颈进行底层重构，优化 MCTS 并行策略与数据流转效率。 |
| **2025-11-23** | [**引入注意力机制，亚马逊棋Bot神经网络架构再颠覆**](https://ly-geming.github.io/2025/11/23/%E5%BC%95%E5%85%A5%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%EF%BC%8C%E4%BA%9A%E9%A9%AC%E9%80%8A%E6%A3%8Bbot%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%9E%B6%E6%9E%84%E5%86%8D%E9%A2%A0%E8%A6%86/) | **[核心技术]** 详解如何设计 Dynamic Feature Update Head，利用 Attention 解决“移动+射箭”联合动作空间的稀疏性难题。 |
| **2025-11-24** | [**互动网页：全面理解我是如何构建属于亚马逊棋的AlphaZero的**](https://ly-geming.github.io/2025/11/24/%E7%9B%B4%E8%A7%82%E7%90%86%E8%A7%A3/) | 通过直观的交互式演示，解构模型的决策逻辑与构建过程。 |

---

## ✨ 核心特性 (Key Features)

* **🚀 高性能 C++ 扩展** 底层游戏逻辑 (`GameCore`) 和特征提取 (`FeatureExtractor`) 采用 C++ 位运算（Bitboard）实现，大幅提升 MCTS 模拟速度。

* **🧠 动态特征更新网络** 摒弃传统的 Flatten 输出，采用 ResNet + Attention 架构，在隐空间中动态模拟“移动后射箭”的价值流转。

* **🔄 并行自对弈** 支持多进程并行 MCTS 搜索，配合 GPU 批量推理，最大化硬件利用率。

* **💾 高效经验回放** C++ 实现的线程安全 Replay Buffer，支持大规模训练数据的无锁存取。

---

## 📂 项目结构 (Project Structure)

```bash
.
├── amazons/                         # Python 核心逻辑
│   ├── AmazonsGame.py
│   ├── pytorch/
│   │   ├── AmazonsPytorch.py
│   │   └── NNet.py
│   └── train_config.py
├── amazons_ops.cpp                  # C++ 核心扩展
├── origin.cpp                       # 可能用于原型对比/遗留代码
├── Arena.py                         # 竞技场对战逻辑
├── CONTRIBUTING.md                  # 贡献指南
├── Dispatcher.py                    # 异步/任务调度逻辑
├── Game.py                         # 游戏主逻辑
├── GpuWorker.py                     # GPU 推理工作线程
├── LICENSE                          # MIT 许可证
├── NeuralNet.py                     # 网络接口/训练逻辑封装
├── OrchestratedMCTS.py              # 并行 MCTS 搜索逻辑
├── OrchestratedParallelCoach.py     # Self-Play / Arena 训练调度
├── ProcessManager.py                # 进程管理工具
├── README.md                        # 项目说明文档
├── requirements.txt                 # Python 依赖清单
├── setup.py                        # C++ 扩展编译配置
├── train_distill.py                 # 用于蒸馏 / 训练流程扩展
├── trans.py                         # 支持脚本/数据转换工具
└── utils.py                        # 通用工具函数

```



## 🛠️ 安装与编译 (Installation)

### 环境要求

* Python >= 3.8

* PyTorch >= 1.9.0 (CUDA 推荐)

* C++ 编译器 (GCC/Clang/MSVC) 支持 C++17

### 1. 克隆仓库

```bash
git clone <your-repo-url>
cd V10
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 编译 C++ 高性能扩展 (重要)

这是项目运行的核心，必须首先完成。

```bash
python setup.py build_ext --inplace
```

*编译成功后，目录中应出现 `amazons_ops.pyd` (Windows) 或 `amazons_ops.so` (Linux)。*

---

## 🚀 快速开始 (Quick Start)

### 启动 AlphaZero 自我博弈训练

该命令将启动完整的强化学习循环：自对弈收集数据 -> 神经网络训练 -> 竞技场评估 -> 模型迭代。

```bash
# 推荐使用 load-best 继续优化最佳模型
python amazons/train.py --load-best

# 或者从零开始训练
python amazons/train.py
```

### 模型验证

运行推理验证脚本，检查模型是否理解复杂的“困毙”规则以及 Attention 机制是否正常工作。

```bash
python verify_model_inference.py
```

---

## ⚙️ 核心组件开发 (Development)

### 调用 C++ 特征提取

```python
import amazons_ops
import numpy as np

# 返回 (7, 8, 8) 的浮点特征张量，包含：
# MyPiece, OpPiece, Obstacles, MyMobility, OpMobility, MyShoot, OpShoot
features = amazons_ops.compute_7ch_features(board_my, board_op, board_arr)
```

### 调用 Replay Buffer

```python
buffer = amazons_ops.ReplayBuffer(capacity=1000000)
# 线程安全地添加自对弈样本
buffer.add_sample(board, player, winner, srcs, dsts, arrs, probs)
```

---

## 🙏 致谢 (Acknowledgements)

特别致谢：诚挚感谢我的 **《计算概论A》** 课程的助教。本次大作业不仅为我提供了深入探索神经网络与 AlphaZero 算法的宝贵契机，助教们的悉心指导更在项目实现过程中给予了我巨大的帮助与启发。

* 算法灵感来源于 DeepMind 的 AlphaZero 论文：[Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)

* 感谢开源社区对 AlphaZero General 框架的探索，本项目在此基础上针对亚马逊棋进行了深度定制。

---

## 📜 许可证 (License)

本项目采用 [MIT License](LICENSE) 许可证。
