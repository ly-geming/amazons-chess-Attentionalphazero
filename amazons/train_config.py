# Amazon Chess Training Configuration
# 针对 RTX 4090、ResNet-40 (20 Blocks)、One-Hot 编码及 N-to-1 架构优化

class TrainingConfig:
    def __init__(self, **kwargs):
        # ---------------------------------------------------------------------
        # 1. 神经网络与训练参数 (Neural Network & Training)
        # ---------------------------------------------------------------------
        # [ResNet-40 核心配置]
        self.num_channels = kwargs.get('num_channels', 256)       # 通道数 (配合 ResNet-40 建议 256)
        self.num_res_blocks = kwargs.get('num_res_blocks', 20)      # 残差块数量 (20 blocks = 41 layers)
        self.dropout = kwargs.get('dropout', 0.3)            # 防止过拟合
        
        # [新增] 输入通道数 (默认为 7，适配 Dual Attention 架构)
        # 允许从 kwargs 读取，如果没有则默认为 7
        self.input_channels = kwargs.get('input_channels', 7)

        # [训练 Batch Size]
        # 4090 显存大，512 能让 GPU 跑满，训练更稳定
        self.training_batch_size = kwargs.get('training_batch_size', 512)  
        
        # 优化器参数
        self.learning_rate = kwargs.get('learning_rate', 0.001)    # ResNet 较深，0.001 起步比较稳
        self.lr_multiplier = kwargs.get('lr_multiplier', 1.0)      
        self.weight_decay = kwargs.get('weight_decay', 1e-4)      
        self.momentum = kwargs.get('momentum', 0.9)           
        self.epochs = kwargs.get('epochs', 5)             

        # ---------------------------------------------------------------------
        # 2. MCTS 搜索参数 (MCTS)
        # ---------------------------------------------------------------------
        self.num_mcts_sims = kwargs.get('num_mcts_sims', 1000)      # 标准 AlphaZero 设置
        self.cpuct = kwargs.get('cpuct', 1.5)           # 探索系数
        
        # [狄利克雷噪声] AlphaZero 自对弈探索的关键
        self.dirichlet_alpha = kwargs.get('dirichlet_alpha', 0.02)    # 8x8 棋盘动作空间较大，0.3 比较合适
        self.dirichlet_epsilon = kwargs.get('dirichlet_epsilon', 0.25) # 噪声占比 25%

        self.temp_threshold = kwargs.get('temp_threshold', 8)      # 前 15 步采用概率采样，之后采用 Argmax

        # ---------------------------------------------------------------------
        # 3. 系统与推理架构参数 (System & Architecture)
        # ---------------------------------------------------------------------
        self.use_n_to_1_architecture = kwargs.get('use_n_to_1_architecture', True)
        
        # Actor 数量：建议设为 CPU 物理核心数
        # 假设你是 i9，跑 32 个 Actor 没问题
        self.num_actors = kwargs.get('num_actors', 32)       
        
        # [推理 Batch Size]
        # 决定了 Dispatcher 积攒多少个请求才发给 GPU。
        # 4090 吞吐量极大，太小的 batch (如 16) 会导致 CPU 瓶颈
        self.inference_batch_size = kwargs.get('inference_batch_size', 128) 
        
        # 调度器数量：1 个足以处理每秒 10万+ 请求
        self.num_dispatchers = kwargs.get('num_dispatchers', 1)      
        
        self.use_gpu = kwargs.get('use_gpu', True)
        self.board_size = kwargs.get('board_size', 8)
        
        # 迭代参数
        self.num_iters = kwargs.get('num_iters', 1)           # 由 starttrain.py 控制循环，这里设 1 即可
        self.num_eps = kwargs.get('num_eps', 128)           # 每轮自对弈局数
        self.maxlenOfQueue = kwargs.get('maxlenOfQueue', 500000)  # Replay Buffer 大小
        self.numItersForTrainExamplesHistory = kwargs.get('numItersForTrainExamplesHistory', 50) # 保留最近 20 轮的历史数据

        # 竞技场参数
        self.arena_compare = kwargs.get('arena_compare', 40)       # 竞技场对局数 (ResNet 较深，推理慢，40局比较平衡)
        self.update_threshold = kwargs.get('update_threshold', 0.55)  # 胜率超过 55% 更新模型
        self.arena_batch_size = kwargs.get('arena_batch_size', 32)    # 竞技场推理 Batch


        # 路径配置
        self.checkpoint = kwargs.get('checkpoint', './checkpoint/')
        self.load_model = kwargs.get('load_model', False)
        self.load_folder_file = kwargs.get('load_folder_file', ('./checkpoint/', 'best.pth.tar'))

        # 兼容性更新
        self._update_compat()

    def _update_compat(self):
        """更新旧代码使用的驼峰命名变量"""
        self.numIters = self.num_iters
        self.numEps = self.num_eps
        self.numMCTSSims = self.num_mcts_sims
        self.tempThreshold = self.temp_threshold
        self.updateThreshold = self.update_threshold
        self.arenaCompare = self.arena_compare
        self.numChannels = self.num_channels
        self.learningRate = self.learning_rate
        self.loadFolderFile = self.load_folder_file
        self.loadModel = self.load_model
        self.numActors = self.num_actors
        self.boardSize = self.board_size
        self.batch_size = self.training_batch_size 
        # 新增兼容
        self.inputChannels = self.input_channels


# 1. 快速测试配置 (用于跑通流程)
class FastTrainingConfig(TrainingConfig):
    def __init__(self, **kwargs):
        defaults = {
            'num_res_blocks': 2,       # 极浅网络
            'num_channels': 64,
            'num_mcts_sims': 25,
            'num_eps': 4,
            'num_iters': 1,
            'num_actors': 4,
            'arena_compare': 4,
            'training_batch_size': 16,
            'inference_batch_size': 4
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


# 2. 长期训练配置 (生产环境 - 高质量)
class LongTermTrainingConfig(TrainingConfig):
    def __init__(self, **kwargs):
        defaults = {
            'num_res_blocks': 20,
            'num_channels': 256,
            'learning_rate': 0.0005,   # 降低学习率求稳
            'num_mcts_sims': 1600,     # 增加搜索深度
            'cpuct': 1.25,
            'num_eps': 200,            # 增加每轮局数
            'arena_compare': 60,       # 更严谨的竞技场
            'update_threshold': 0.55,
            'training_batch_size': 512,
            'inference_batch_size': 128,
            'num_actors': 32
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


# 3. 标准 N-to-1 配置 (默认推荐 - 4090 平衡版)
# 修改 NTo1TrainingConfig 的默认值
class NTo1TrainingConfig(TrainingConfig):
    def __init__(self, **kwargs):
        defaults = {
            'num_mcts_sims': 10,       # Self-Play 用 800 就够了 (配合 Acc5 84%)
            'num_res_blocks': 20,
            'num_channels': 256,
            'training_batch_size': 256, # [修改] 降到 256，配合 train_distill 的优化
            'inference_batch_size': 64,
            'num_actors': 32,
            'num_dispatchers': 1,       # 1个足够
            'input_channels': 7,        # [关键]
            'learning_rate': 2e-4       # [修改] 2e-4 (FP32 Stable LR)
        }
        defaults.update(kwargs)
        super().__init__(**defaults)