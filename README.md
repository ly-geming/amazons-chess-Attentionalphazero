# Amazons Chess AlphaZero Implementation

一个基于 AlphaZero 算法的亚马逊棋（Amazons Chess）AI 实现，使用 PyTorch 和 C++ 扩展进行高性能训练和推理。

## 特性

- 🚀 **高性能 C++ 扩展**：使用位运算优化的特征提取，加速训练和推理
- 🧠 **AlphaZero MCTS**：实现完整的蒙特卡洛树搜索算法
- 💾 **经验回放缓冲**：线程安全的数据缓冲池，支持大规模训练数据管理
- 🔄 **并行训练**：支持多进程自对弈和分布式训练
- 📊 **7通道特征提取**：包含棋盘状态、移动性、射击潜力等多维特征

## 项目结构

```
.
├── amazons/              # 核心游戏逻辑和神经网络
│   ├── AmazonsGame.py    # 游戏规则实现
│   ├── pytorch/          # PyTorch 神经网络定义
│   └── train_config.py   # 训练配置
├── amazons_ops.cpp       # C++ 扩展（特征提取和数据缓冲）
├── OrchestratedMCTS.py   # MCTS 实现
├── OrchestratedParallelCoach.py  # 并行训练协调器
├── GpuWorker.py          # GPU 推理工作进程
├── train_distill.py      # 蒸馏训练主程序
└── setup.py              # C++ 扩展编译配置
```

## 环境要求

- Python >= 3.7
- PyTorch >= 1.9.0
- C++ 编译器（Windows: MSVC，Linux: GCC）
- CUDA（可选，用于 GPU 训练）

## 安装

1. 克隆仓库：
```bash
git clone <your-repo-url>
cd V10
```

2. 安装 Python 依赖：
```bash
pip install -r requirements.txt
```

3. 编译 C++ 扩展：
```bash
python setup.py build_ext --inplace
```

这将生成 `amazons_ops.pyd`（Windows）或 `amazons_ops.so`（Linux）文件。

## 快速开始

### 训练模型

```bash
# 使用蒸馏训练（推荐）
python train_distill.py

# 或使用并行训练
python amazons/train.py
```

### 测试模型

```bash
cd amazons
python test.py
```

## 核心组件

### C++ 扩展模块 (`amazons_ops`)

- **特征提取**：`compute_7ch_features()` - 生成7通道棋盘特征
- **经验缓冲**：`ReplayBuffer` 类 - 线程安全的数据管理

```python
import amazons_ops

# 特征提取
features = amazons_ops.compute_7ch_features(board_my, board_op, board_arr)

# 创建经验缓冲
buffer = amazons_ops.ReplayBuffer(capacity=1000000)
buffer.add_sample(board, player, winner, srcs, dsts, arrs, probs)
batch = buffer.get_batch(batch_size=256)
```

### 神经网络架构

- ResNet 主干网络
- 多头输出：移动概率、射击概率、价值估计
- 支持混合精度训练

### MCTS 算法

- UCT 选择策略
- 并行树搜索
- 支持 GPU 加速推理

## 训练配置

训练参数可在 `amazons/train_config.py` 中配置：

- `FastTrainingConfig` - 快速测试配置
- `NTo1TrainingConfig` - 标准训练配置
- `LongTermTrainingConfig` - 长期训练配置

## 数据格式

训练数据以二进制格式存储（`.bin` 文件），包含：
- 棋盘状态（7通道特征）
- 动作分布（源位置、目标位置、箭位置、概率）
- 游戏结果（价值标签）

## 性能优化

- C++ 位运算优化特征计算
- 批量推理减少 GPU 调用开销
- 多进程并行自对弈
- 经验回放缓冲复用数据

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 贡献

欢迎提交 Issue 和 Pull Request！

## 致谢

本项目基于 AlphaZero 算法实现，参考了以下资源：
- [AlphaZero General](https://github.com/suragnair/alpha-zero-general)
- DeepMind AlphaZero 论文

