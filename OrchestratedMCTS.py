import math
import numpy as np
import time
import queue
import logging
from multiprocessing import shared_memory

EPS = 1e-8
log = logging.getLogger(__name__)

class OrchestratedMCTS():
    def __init__(self, game, args, actor_id=0, request_queue=None, response_queue=None, shm_info=None):
        self.game = game
        self.args = args
        self.actor_id = actor_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}
        
        # [Shared Memory Init]
        self.shm = None
        self.shared_board_buffer = None
        self.use_shm = False

        if shm_info:
            try:
                self.shm = shared_memory.SharedMemory(name=shm_info['name'])
                # 映射为 numpy 数组，形状 (N, 8, 8)
                self.shared_board_buffer = np.ndarray(
                    (shm_info['num_actors'], shm_info['shape'][0], shm_info['shape'][1]),
                    dtype=np.int32,
                    buffer=self.shm.buf
                )
                self.use_shm = True
            except Exception as e:
                log.error(f"Actor {actor_id} Failed to attach SHM: {e}")
                self.use_shm = False

    def close_shm(self):
        if self.shm:
            try:
                self.shm.close()
            except:
                pass
            self.shm = None

    def clear_tree(self):
        self.Qsa.clear(); self.Nsa.clear(); self.Ns.clear(); self.Ps.clear(); self.Es.clear(); self.Vs.clear()

    def getActionProb(self, canonicalBoard, temp=1):
        if self.args.numMCTSSims <= 0:
             raise ValueError("numMCTSSims must be > 0")

        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard, depth=0)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
        else:
            counts = [x ** (1. / temp) for x in counts]
            counts_sum = float(sum(counts))
            if counts_sum == 0:
                probs = [1.0 / len(counts)] * len(counts)
            else:
                probs = [x / counts_sum for x in counts]

        return probs

    def search(self, canonicalBoard, depth=0, max_depth=100):
        if depth >= max_depth: return 0
        
        canonicalBoard = np.ascontiguousarray(canonicalBoard, dtype=np.int32)

        s = self.game.stringRepresentation(canonicalBoard)
        if s not in self.Es: self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0: return -self.Es[s]

        if s not in self.Ps:
            # [IPC] 获取预测
            # pi_pack: ((p_move, p_arrow), v)
            pi_pack, v = self._get_prediction_from_shared_evaluator(canonicalBoard)
            if isinstance(v, np.ndarray): v = v.item()
            
            # [关键修改] 解包 Model B 输出
            p_move, p_arrow = pi_pack[0], pi_pack[1]

            # C++ 极速同步状态以获取合法动作
            try:
                self.game.game.set_board(canonicalBoard, 1)
            except AttributeError:
                pass
            
            legal_moves = self.game.game.getLegalMoves(1)
            prob_vector = np.zeros(self.game.getActionSize())
            
            limit = min(len(legal_moves), self.game.getActionSize())
            for idx in range(limit):
                move = legal_moves[idx]
                s_idx = move.from_pos.row * 8 + move.from_pos.col
                d_idx = move.to_pos.row * 8 + move.to_pos.col
                a_idx = move.arrow_pos.row * 8 + move.arrow_pos.col
                
                # [关键修改] 计算联合概率 P(Move) * P(Arr|Move)
                move_joint_idx = s_idx * 64 + d_idx
                
                prob_move_part = p_move[move_joint_idx]
                prob_arrow_part = p_arrow[d_idx][a_idx] # 注意这里使用的是 p_arrow[d_idx] (近似) 或全量
                # 纠正：GpuWorker 传回的 p_arrow 是 (4096, 64)
                # 所以应该是 p_arrow[move_joint_idx][a_idx]
                # 或者是 (64, 64) 的 Matrix? 
                # 查看 GpuWorker.py: p_arrow = torch.exp(log_p_arrow).data.cpu().numpy() -> (B, 4096, 64)
                # 所以这里应该是:
                prob_arrow_part = p_arrow[move_joint_idx][a_idx]
                
                prob = prob_move_part * prob_arrow_part
                prob_vector[idx] = prob

            sum_probs = np.sum(prob_vector)
            if sum_probs > 1e-8: prob_vector /= sum_probs
            else: 
                if limit > 0: prob_vector[:limit] = 1.0 / limit
            
            # Dirichlet Noise
            if depth == 0 and hasattr(self.args, 'dirichlet_alpha'):
                alpha = self.args.dirichlet_alpha
                epsilon = self.args.dirichlet_epsilon
                if limit > 0:
                    noise = np.random.dirichlet([alpha] * limit)
                    prob_vector[:limit] = (1 - epsilon) * prob_vector[:limit] + epsilon * noise

            self.Ps[s] = prob_vector
            self.Vs[s] = self.game.getValidMoves(canonicalBoard, 1)
            self.Ns[s] = 0
            
            # [关键] 直接返回 -v (负值)，因为这个 v 是当前玩家的胜率
            # 而 search 函数返回的是对父节点(对手)的价值
            return -v 

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa: u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else: u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
                if u > cur_best: cur_best = u; best_act = a
        
        if best_act == -1: return -1
        a = best_act
        
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)
        
        v = self.search(next_s, depth + 1, max_depth)
        
        # [关键] 更新 Q 值：这里 v 已经是相对于当前玩家的对手的价值
        # 所以对当前玩家来说，收益是 -v
        # AlphaZero 标准公式: Q = (N*Q + v) / (N+1)
        # 但这里的 v 是递归返回上来的，通常在 search 内部做了取反
        # 让我们理一下：
        # search(next_s) 返回的是 next_player 的对手(也就是当前玩家)的价值吗？
        # search 定义: return value from perspective of player invoking search? NO.
        # search(s) returns value for player at state s.
        # So search(next_s) returns value for next_player.
        # We are current_player. We want to maximize our value, which is -search(next_s).
        # So v_for_me = -v.
        
        v_for_me = -v
        
        if (s, a) in self.Qsa: self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v_for_me) / (self.Nsa[(s, a)] + 1); self.Nsa[(s, a)] += 1
        else: self.Qsa[(s, a)] = v_for_me; self.Nsa[(s, a)] = 1
        self.Ns[s] += 1
        
        # 返回当前状态对上一层(父节点)的价值，即 -v_for_me
        return -v_for_me

    def _get_prediction_from_shared_evaluator(self, canonicalBoard):
        if self.request_queue is not None and self.response_queue is not None:
            try:
                if self.use_shm and self.shared_board_buffer is not None:
                    self.shared_board_buffer[self.actor_id] = canonicalBoard
                    self.request_queue.put(self.actor_id)
                else:
                    self.request_queue.put((self.actor_id, canonicalBoard))
                
                return self.response_queue.get()
            except Exception as e:
                log.error(f"Actor {self.actor_id} IPC Error: {e}")
                raise e
        raise RuntimeError("Actor queues not initialized")