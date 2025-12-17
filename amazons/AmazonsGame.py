from __future__ import print_function
import sys
import numpy as np
import os
sys.path.append('..')
from Game import Game

# Add the current directory to Python path to find the amazons_engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import amazons_engine
    engine_available = True
except ImportError as e:
    print(f"Warning: Could not import amazons_engine: {e}")
    engine_available = False

class AmazonsGame(Game):
    """
    Amazon Chess implementation for AlphaZero General framework
    Backed by high-performance C++ Engine
    """

    def __init__(self, n=8):
        self.n = n
        if engine_available:
            # [安全检查] 防止维度不匹配导致的底层内存错乱
            # amazoncore.h 中定义了 const int BOARD_SIZE
            if n != amazons_engine.BOARD_SIZE:
                raise ValueError(f"C++ Engine compiled for board size {amazons_engine.BOARD_SIZE}, "
                                 f"but Python requested size {n}. Please recompile engine or change config.")
            
            self.game = amazons_engine.AmazonGame()
            # 预分配临时对象，复用内存，避免频繁 new/delete
            self.temp_game = amazons_engine.AmazonGame()
        else:
            raise ImportError("Amazon engine not available. Please compile amazons_engine.cpp.")

    def _sync_board(self, game_instance, board, player):
        """
        [核心安全函数] 将 Python Numpy 数组安全地同步到 C++ 引擎
        
        Rationale:
        1. np.rot90 等操作返回的是视图 (View)，内存可能不连续。传给 C++ 指针会导致读取错位。
        2. 必须强制转换为 int32，匹配 C++ 的 int。
        """
        # ascontiguousarray 确保内存连续，dtype=int32 确保字节宽度正确
        board_c = np.ascontiguousarray(board, dtype=np.int32)
        game_instance.set_board(board_c, player)

    def getInitBoard(self):
        self.game.initializeBoard()
        return np.array(self.game.getBoard(), dtype=np.int32)

    def getBoardSize(self):
        return (self.n, self.n)

    def getCurrentPlayer(self):
        return self.game.getCurrentPlayer()

    def getActionSize(self):
        # 亚马逊棋合法动作极多，这里定义一个足够大的 buffer size
        # MCTS 会根据 getValidMoves 返回的实际长度进行截断处理
        return 5000

    def encode_action(self, move, legal_moves):
        try:
            action_idx = legal_moves.index(move)
            return action_idx
        except ValueError:
            return -1

    def decode_action(self, action_idx, legal_moves):
        if 0 <= action_idx < len(legal_moves):
            return legal_moves[action_idx]
        else:
            return None

    def getNextState(self, board, player, action):
        """
        输入:
            board: 当前棋盘
            player: 当前玩家
            action: 动作在 legal_moves 列表中的索引 (int)
        输出:
            (next_board, next_player)
        """
        # 使用安全同步将状态写入 temp_game
        self._sync_board(self.temp_game, board, player)

        # 获取当前所有合法动作
        legal_moves = self.temp_game.getLegalMoves(player)
        
        # [严格检查] 防止无效动作传入引擎
        if action == -1:
            raise ValueError(f"CRITICAL: getNextState received action=-1 for Player {player}. "
                             "Logic error in MCTS or Action Selection.")
        
        if action >= len(legal_moves):
             raise ValueError(f"CRITICAL: getNextState received out-of-bounds action {action}. "
                              f"Max legal moves: {len(legal_moves)}")

        # 解码动作
        selected_move = self.decode_action(action, legal_moves)

        if selected_move is not None:
            # 执行动作
            move_success = self.temp_game.makeMove(selected_move)
            if move_success:
                next_player = self.temp_game.getCurrentPlayer()
                return (np.array(self.temp_game.getBoard(), dtype=np.int32), next_player)
            else:
                # 理论上 decode 出来的动作必定合法，若引擎拒绝说明状态同步有问题
                raise RuntimeError(f"Engine rejected a validated move at index {action}. Board sync error?")
        else:
            raise ValueError(f"decode_action returned None for index {action}")

    def getValidMoves(self, board, player):
        """
        返回一个固定长度的二进制向量，表示哪些 action_idx 是合法的
        """
        self._sync_board(self.game, board, player)

        legal_moves = self.game.getLegalMoves(player)
        valids = [0] * self.getActionSize()
        
        count = len(legal_moves)
        if count > 0:
            # 标记前 count 个动作为合法
            # 具体的动作含义由 legal_moves[i] 定义
            limit = min(count, self.getActionSize())
            valids[:limit] = [1] * limit

        return np.array(valids, dtype=np.int32)

    def getGameEnded(self, board, player):
        """
        返回: 0 (未结束), 1 (player赢), -1 (player输)
        注意：亚马逊棋中，无路可走即为输
        """
        self._sync_board(self.game, board, player)

        # C++ 引擎的 isGameOver 状态可能因为 set_board 而被重置
        # 所以这里主要依赖 hasLegalMoves 判断
        if not self.game.hasLegalMoves(player):
             return -1 # 当前玩家无子可动，判负
        
        # 如果当前玩家能动，还要检查对手是否能动？
        # 不需要，因为 getGameEnded 是在 player 刚走完或者轮到 player 时调用的
        # 只要轮到的人能走，游戏就没结束
        return 0

    def getCanonicalForm(self, board, player):
        """
        与 Steal2 蒸馏系统保持一致的视角规范：
        Player 1 (White, Bottom): 旋转 180 度到顶部
        Player -1 (Black, Top): 不旋转位置，只反转数值
        """
        if player == 1:
            return np.rot90(board, 2)
        else:
            canonical_board = np.copy(board)
            canonical_board[board == 1] = -1
            canonical_board[board == -1] = 1
            return canonical_board

    def canonical_to_real_action(self, canonical_action_idx, player, real_board):
        """
        [修正后] 适配 TOP-DOWN 视角的动作映射
        由于 getCanonicalForm 中所有玩家都旋转了180度，所以都需要旋转回来
        """
        # 1. 获取 Canonical 视角下的所有动作
        can_board = self.getCanonicalForm(real_board, player)
        self._sync_board(self.game, can_board, 1)
        can_legal_moves = self.game.getLegalMoves(1)
        
        if canonical_action_idx < 0 or canonical_action_idx >= len(can_legal_moves):
            return -1 
            
        can_move = can_legal_moves[canonical_action_idx]

        # 2. 坐标变换逻辑
        # Player 1 需要旋转回来，Player -1 保持原样（与 Steal2 保持一致）
        n = self.n
        
        if player == 1:
            def transform_pos(p):
                return amazons_engine.Position(n - 1 - p.row, n - 1 - p.col)
        else:
            def transform_pos(p):
                return p

        real_move_obj = amazons_engine.Move(
            transform_pos(can_move.from_pos),
            transform_pos(can_move.to_pos),
            transform_pos(can_move.arrow_pos),
            player 
        )

        # 3. 查找真实动作 (代码保持不变)
        self._sync_board(self.game, real_board, player)
        real_legal_moves = self.game.getLegalMoves(player)
        
        try:
            real_action_idx = real_legal_moves.index(real_move_obj)
            return real_action_idx
        except ValueError:
            # [回退机制] 如果对象查找失败，手动比较坐标值
            # 这是为了防止 C++ 对象在 Python 端被重新封装导致 identity 丢失
            for i, m in enumerate(real_legal_moves):
                if (m.from_pos.row == real_move_obj.from_pos.row and 
                    m.from_pos.col == real_move_obj.from_pos.col and
                    m.to_pos.row == real_move_obj.to_pos.row and 
                    m.to_pos.col == real_move_obj.to_pos.col and
                    m.arrow_pos.row == real_move_obj.arrow_pos.row and 
                    m.arrow_pos.col == real_move_obj.arrow_pos.col):
                    return i
            
            print(f"Error: Move mapping failed. Player {player}, CanIdx {canonical_action_idx}")
            return -1

    def getSymmetries(self, board, pi):
        """
        数据增强：旋转和翻转
        注意：pi 在这里是动作索引列表，很难直接旋转。
        如果 Coach 负责处理 Policy Map 的旋转 (OrchestratedParallelCoach 确实是这样做的)，
        这里只需返回 Board 的旋转即可。
        """
        l = []
        for rot in range(4):
            newB = np.rot90(board, rot)
            # 由于 pi 是 index 无法直接旋转，这里填 None 或 原始 pi
            # 实际上 OrchestratedParallelCoach 会忽略这里的 pi，它有自己的逻辑
            l.append((newB, pi))
            
            flipB = np.fliplr(newB)
            l.append((flipB, pi))
        return l

    def stringRepresentation(self, board):
        # 用于 MCTS 哈希，必须高效
        return board.tobytes()

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n): print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")
            for x in range(n):
                piece = board[y][x]
                if piece == 1: print("W ", end="")
                elif piece == -1: print("B ", end="")
                elif piece == 2: print("X ", end="") # 假设 2 是 Arrow
                else: print(". ", end="")
            print("|")
        print("-----------------------")