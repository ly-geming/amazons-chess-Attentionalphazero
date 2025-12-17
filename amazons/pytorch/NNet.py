import os
import sys
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from AmazonsPytorch import AmazonsNNet

try:
    import amazons_ops
    CPP_FEATURE_AVAILABLE = True
except ImportError:
    CPP_FEATURE_AVAILABLE = False

def encode_batch_one_hot(boards_numpy):
    if not isinstance(boards_numpy, np.ndarray):
        boards_numpy = np.array(boards_numpy)
    layer_my = (boards_numpy == 1).astype(np.float32)
    layer_op = (boards_numpy == -1).astype(np.float32)
    layer_arr = (~np.isin(boards_numpy, [0, 1, -1])).astype(np.float32)
    return np.stack([layer_my, layer_op, layer_arr], axis=1)

def get_7ch_features(board_3ch):
    """
    统一接口：优先使用 C++ 版本
    """
    if CPP_FEATURE_AVAILABLE:
        # C++ 接口需要分离为 my, op, arr 输入
        B = board_3ch.shape[0]
        board_7ch = np.zeros((B, 7, 8, 8), dtype=np.float32)
        for i in range(B):
            my_map = board_3ch[i, 0].astype(np.int32)
            op_map = board_3ch[i, 1].astype(np.int32)
            arr_map = board_3ch[i, 2].astype(np.int32)
            board_7ch[i] = amazons_ops.compute_7ch_features(my_map, op_map, arr_map)
        return board_7ch
    else:
        # 简单的 Python 回退 (仅做示意，建议务必编译 C++)
        log.warning("Falling back to dummy features (compile C++ for speed!)")
        B = board_3ch.shape[0]
        res = np.zeros((B, 7, 8, 8), dtype=np.float32)
        res[:, :3] = board_3ch
        return res

class NNetWrapper(NeuralNet):
    def __init__(self, game, args):
        self.nnet = AmazonsNNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
        self.nnet.to(self.device)
        self.optimizer = optim.AdamW(self.nnet.parameters(), lr=args.learningRate, weight_decay=1e-4)

    def train(self, examples):
        """
        注意：现在主要的训练逻辑在 train_distill.py 中。
        这个函数保留用于兼容旧的 Coach 框架，但建议使用 train_distill.py。
        """
        log.warning("Please use train_distill.py for training with Teacher Distillation!")
        pass

    def predict(self, board):
        """
        推理接口：供 Python 端 MCTS 使用
        返回: (P(Move), P(Arrow|Dst), v) 
        或者处理成 MCTS 需要的格式
        """
        try:
            # 1. 预处理
            board_batch = board[np.newaxis, :, :]
            board_encoded = encode_batch_one_hot(board_batch)
            board_encoded = get_7ch_features(board_encoded)
            board_tensor = torch.FloatTensor(board_encoded).to(self.device)
            
            self.nnet.eval()
            with torch.no_grad():
                # [关键修改] 接收 3 个返回值，而不是旧的 4 个
                log_p_move, log_p_arrow, v = self.nnet(board_tensor)

            # 2. 转 Numpy
            # p_move: (64, 64) matrix of P(src, dst)
            p_move = torch.exp(log_p_move).data.cpu().numpy()[0].reshape(64, 64)
            # p_arrow: (64, 64) matrix of P(arr | dst)
            p_arrow = torch.exp(log_p_arrow).data.cpu().numpy()[0] 
            value = v.data.cpu().numpy()[0]

            # 返回给 MCTS 使用
            # 注意：标准的 MCTS 可能只需要一个 pi 向量。
            # 这里返回原始矩阵，需要 MCTS 端适配，或者在这里做后处理。
            return (p_move, p_arrow), value

        except Exception as e:
            log.error(f"Predict Error: {e}")
            raise e

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder): os.mkdir(folder)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath): raise FileNotFoundError(f"No model in {filepath}")
        map_location = self.device
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])