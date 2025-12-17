import logging
import os
import sys
import multiprocessing as mp
from multiprocessing import shared_memory
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import numpy as np
from tqdm import tqdm
import time
import gc
import queue
import copy 
import torch
import torch.optim as optim
import torch.nn.functional as F

from Arena import Arena

log = logging.getLogger(__name__)

# =============================================================================
# Global Coordination & Helpers
# =============================================================================

COORDINATION_DATA = {
    'shared_request_queue': None,
    'actor_response_queues': [],
    'shm_name': None,
    'num_actors': 0,
    'board_shape': (8, 8)
}

ARENA_COORDINATION = {
    'new_req_queue': None,
    'new_resp_queues': [],
    'best_req_queue': None,
    'best_resp_queues': []
}

def single_arg_wrapper_for_imap(args_tuple):
    return execute_episode_with_shared_evaluator_standalone(*args_tuple[0])

def initialize_worker(shared_request_queue, actor_response_queues, shm_name, num_actors, board_n):
    global COORDINATION_DATA
    COORDINATION_DATA['shared_request_queue'] = shared_request_queue
    COORDINATION_DATA['actor_response_queues'] = actor_response_queues
    COORDINATION_DATA['shm_name'] = shm_name
    COORDINATION_DATA['num_actors'] = num_actors
    COORDINATION_DATA['board_shape'] = (board_n, board_n)

def initialize_arena_worker(new_req_queue, new_resp_queues, best_req_queue, best_resp_queues):
    global ARENA_COORDINATION
    ARENA_COORDINATION['new_req_queue'] = new_req_queue
    ARENA_COORDINATION['new_resp_queues'] = new_resp_queues
    ARENA_COORDINATION['best_req_queue'] = best_req_queue
    ARENA_COORDINATION['best_resp_queues'] = best_resp_queues

class AverageMeter(object):
    def __init__(self): self.reset()
    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def rotate_tuple(src, dst, arr, k, n=8):
    """
    旋转坐标，与 np.rot90 的顺时针旋转保持一致
    np.rot90(board, k) 是顺时针旋转 k*90度
    对应的坐标变换：顺时针旋转 k*90度
    公式: (r, c) -> (n-1-c, r)
    """
    def rot_point(idx, k_rot):
        r, c = idx // n, idx % n
        # 顺时针旋转：k=1: (r,c) -> (n-1-c, r)
        # k=2: (r,c) -> (n-1-r, n-1-c)  
        # k=3: (r,c) -> (c, n-1-r)
        for _ in range(k_rot):
            r, c = n - 1 - c, r
        return r * n + c
    return rot_point(src, k), rot_point(dst, k), rot_point(arr, k)

def flip_tuple(src, dst, arr, n=8):
    def flp_point(idx):
        r, c = idx // n, idx % n; c = n - 1 - c
        return r * n + c
    return flp_point(src), flp_point(dst), flp_point(arr)

def calculate_metrics(log_p_move, log_p_arrow, srcs, dsts, arrs, probs):
    move_flat_indices = srcs * 64 + dsts
    log_p_arrow = log_p_arrow.squeeze(1)
    
    selected_move_logp = log_p_move[torch.arange(log_p_move.size(0)), move_flat_indices]
    selected_arrow_logp = log_p_arrow[torch.arange(log_p_move.size(0)), arrs]
    model_logp = selected_move_logp + selected_arrow_logp
    
    cross_entropy_sum = -torch.sum(probs * model_logp)
    
    with torch.no_grad():
        pred_move = log_p_move.argmax(dim=1)
        is_match = (pred_move == move_flat_indices)
        weighted_hits = (is_match.float() * probs).sum()
        
        _, top5_indices = log_p_move.topk(5, dim=1)
        is_match_5 = (top5_indices == move_flat_indices.unsqueeze(1)).any(dim=1)
        weighted_hits_5 = (is_match_5.float() * probs).sum()
        
        total_prob_mass = probs.sum()

    return cross_entropy_sum, weighted_hits, weighted_hits_5, total_prob_mass

# =============================================================================
# Self-Play Logic
# =============================================================================

def execute_episode_with_shared_evaluator_standalone(actor_id, game_class_name, game_params, mcts_params, network_params):
    if game_class_name == "AmazonsGame":
        from amazons.AmazonsGame import AmazonsGame as GameClass
    else: raise ValueError(f"Unknown game class: {game_class_name}")

    game = GameClass(**game_params)
    shared_request_queue = COORDINATION_DATA['shared_request_queue']
    actor_response_queue = COORDINATION_DATA['actor_response_queues'][actor_id]
    
    shm_info = None 

    from OrchestratedMCTS import OrchestratedMCTS
    mcts = OrchestratedMCTS(game, mcts_params, actor_id=actor_id,
                           request_queue=shared_request_queue, response_queue=actor_response_queue,
                           shm_info=shm_info)
    mcts.clear_tree()
    
    import random
    time.sleep(random.uniform(0.1, 3.0))

    trainExamples = []
    board = game.getInitBoard()
    curPlayer = 1
    episodeStep = 0

    try:
        while True:
            episodeStep += 1
            mcts.clear_tree() 
            canonicalBoard = game.getCanonicalForm(board, curPlayer)
            pi_for_training = mcts.getActionProb(canonicalBoard, temp=1)
            temp_for_action = int(episodeStep < mcts_params.tempThreshold)
            
            policy_tuples = []
            board_c = np.ascontiguousarray(canonicalBoard, dtype=np.int32)
            game.game.set_board(board_c, 1)
            legal_moves = game.game.getLegalMoves(1)
            
            limit = min(len(legal_moves), len(pi_for_training))
            for idx in range(limit):
                prob = pi_for_training[idx]
                if prob > 1e-8:
                    move = legal_moves[idx]
                    s = move.from_pos.row * 8 + move.from_pos.col
                    d = move.to_pos.row * 8 + move.to_pos.col
                    a = move.arrow_pos.row * 8 + move.arrow_pos.col
                    policy_tuples.append((s, d, a, prob))

            syms = []
            syms.append((canonicalBoard, policy_tuples))
            for k in range(1, 4):
                b_rot = np.rot90(canonicalBoard, k)
                p_rot = [(*rotate_tuple(s, d, a, k), p) for s, d, a, p in policy_tuples]
                syms.append((b_rot, p_rot))
            b_flip = np.fliplr(canonicalBoard)
            p_flip = [(*flip_tuple(s, d, a), p) for s, d, a, p in policy_tuples]
            syms.append((b_flip, p_flip))
            for k in range(1, 4):
                b_rot = np.rot90(b_flip, k)
                p_rot = [(*rotate_tuple(s, d, a, k), p) for s, d, a, p in p_flip]
                syms.append((b_rot, p_rot))

            for b, p_list in syms:
                trainExamples.append([b, curPlayer, p_list, None])

            if temp_for_action == 0: action_canonical = np.argmax(pi_for_training)
            else: action_canonical = np.random.choice(len(pi_for_training), p=pi_for_training)

            action_real = game.canonical_to_real_action(action_canonical, curPlayer, board)
            if action_real == -1: raise RuntimeError("Canonical Action Fail")

            board, curPlayer = game.getNextState(board, curPlayer, action_real)
            r = game.getGameEnded(board, curPlayer)

            if r != 0:
                mcts.clear_tree(); mcts.close_shm() 
                # x[0]: board, x[1]: player, x[2]: policy, x[3]: reward (backfilled here)
                formatted_examples = [(x[0], x[1], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]
                winner = 0 
                if abs(r) > 0.1: winner = 1 if (r * curPlayer) > 0 else -1
                return formatted_examples, winner
                
    except Exception as e:
        log.error(f"Actor {actor_id} Exception: {e}")
        mcts.close_shm() 
        raise e

# =============================================================================
# Arena Logic (Fully Expanded)
# =============================================================================

def execute_arena_game_parallel(game_idx, p1_is_new, game_cls_name, game_params, mcts_args):
    """
    Arena Logic: Adapted for Joint Policy Network (Greedy Policy)
    """
    global ARENA_COORDINATION
    import numpy as np
    
    if game_cls_name == "AmazonsGame":
        from amazons.AmazonsGame import AmazonsGame
        game = AmazonsGame(**game_params)
    else:
        raise ValueError(f"Unknown game class: {game_cls_name}")

    req_queue_new = ARENA_COORDINATION['new_req_queue']
    resp_queue_new = ARENA_COORDINATION['new_resp_queues'][game_idx]
    req_queue_best = ARENA_COORDINATION['best_req_queue']
    resp_queue_best = ARENA_COORDINATION['best_resp_queues'][game_idx]

    def play_greedy(board, curPlayer_real, req_q, resp_q):
        req_q.put((game_idx, board))
        
        # Wait for GPU Inference result
        pi_pack, _ = resp_q.get()
        p_move, p_arrow = pi_pack[0], pi_pack[1]
        
        # Flatten Move Probs
        p_move = p_move.flatten()
        
        # C++ Sync for legal moves
        board_c = np.ascontiguousarray(board, dtype=np.int32)
        game.game.set_board(board_c, 1)
        legal_moves = game.game.getLegalMoves(1)
        
        if len(legal_moves) == 0:
            return -1

        # [修改] 收集所有候选动作的概率
        probs = []
        candidates = []
        
        # Greedy Selection based on Joint Probability
        for i, move in enumerate(legal_moves):
            s = move.from_pos.row * 8 + move.from_pos.col
            d = move.to_pos.row * 8 + move.to_pos.col
            a = move.arrow_pos.row * 8 + move.arrow_pos.col
            
            joint_idx = s * 64 + d
            # Model B Output Decoding
            prob_move_part = p_move[joint_idx]
            
            # 注意：需确保 p_arrow 维度与此处访问方式匹配 
            # (通常是 [4096, 64] 或 [64*64, 64])
            prob_arrow_part = p_arrow[joint_idx][a] 
            
            # 联合概率 P(Move) * P(Arrow|Move)
            prob = prob_move_part * prob_arrow_part
            
            probs.append(prob)
            candidates.append(i)
        
        probs = np.array(probs, dtype=np.float64)
        
        # [关键修正] 引入微小随机噪声 (Noise Injection)
        # 目的：打破网络输出确定性导致的完全重复对局
        # 这里的 1e-6 足够小，不会改变大概率动作的排序，但足以在概率相等或极其接近时引入随机性
        if len(probs) > 0:
            noise = np.random.normal(0, 1e-6, size=probs.shape)
            best_local_idx = np.argmax(probs + noise)
            best_move_idx_canonical = candidates[best_local_idx]
        else:
            best_move_idx_canonical = -1 # Should be caught by len check above
            
        # [回退保护] 万一还是没选出来 (理论上不可能)
        if best_move_idx_canonical == -1:
            best_move_idx_canonical = np.random.randint(len(legal_moves))
        
        return best_move_idx_canonical
    
    def player_new(board, curPlayer_real, board_real):
        action_canonical = play_greedy(board, curPlayer_real, req_queue_new, resp_queue_new)
        action_real = game.canonical_to_real_action(action_canonical, curPlayer_real, board_real)
        return action_real
    
    def player_best(board, curPlayer_real, board_real):
        action_canonical = play_greedy(board, curPlayer_real, req_queue_best, resp_queue_best)
        action_real = game.canonical_to_real_action(action_canonical, curPlayer_real, board_real)
        return action_real
    
    if p1_is_new:
        p1, p2 = player_new, player_best
    else:
        p1, p2 = player_best, player_new
    
    curPlayer = 1
    board = game.getInitBoard()
    
    step = 0
    while game.getGameEnded(board, curPlayer) == 0:
        step += 1
        if step > 200: return 0, p1_is_new, step, game_idx
        
        canonicalBoard = game.getCanonicalForm(board, curPlayer)
        
        # Current Player move
        # Determine which player function to call based on curPlayer
        # If curPlayer is 1 (Start), and p1_is_new=True, then p1 is New.
        # Players list logic: players[1] -> p1, players[-1] -> p2? No.
        # Let's simplify:
        
        if curPlayer == 1:
            action_real = p1(canonicalBoard, curPlayer, board)
        else:
            action_real = p2(canonicalBoard, curPlayer, board)
            
        valids = game.getValidMoves(board, curPlayer)
        
        if action_real == -1 or valids[action_real] == 0:
            raise RuntimeError(f"Arena: Illegal move {action_real}")
        
        board, curPlayer = game.getNextState(board, curPlayer, action_real)
    
    # Game Ended. Result is 1 (Player 1 wins) or -1 (Player -1 wins)
    # Return value: +1 if New Model wins, -1 if Best Model wins
    # if p1_is_new:
    #    if winner == 1 (P1): result = 1 (New Wins)
    #    if winner == -1 (P2): result = -1 (Best Wins)
    # else:
    #    if winner == 1 (P2/Best): result = -1 (Best Wins)
    #    if winner == -1 (P1/New): result = 1 (New Wins)
    
    raw_result = game.getGameEnded(board, curPlayer) # usually returns 1 or -1 depending on who won?
    # Warning: getGameEnded usually returns result relative to 'player' arg?
    # But here we pass result of game. 
    # Amazons: Last player to move WINS. The one who cannot move LOSES.
    # If loop exits, curPlayer CANNOT move. So curPlayer LOST.
    # Winner is -curPlayer.
    
    winner = -curPlayer
    
    if p1_is_new:
        res = 1 if winner == 1 else -1
    else:
        res = 1 if winner == -1 else -1
        
    return res, p1_is_new, step, game_idx

def wrapper_arena_parallel(args):
    from OrchestratedParallelCoach import execute_arena_game_parallel as exec_arena
    return exec_arena(*args[0]) # <--- 修正：先取 [0] 拿到真正的参数元组，再解包

# =============================================================================
# Coach Class
# =============================================================================

class OrchestratedParallelCoach():
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False
        from ProcessManager import ProcessManager
        self.process_manager = ProcessManager()

    def executeParallelEpisodes(self, num_episodes):
        ctx = mp.get_context('spawn')
        shared_request_queue = ctx.Queue(maxsize=50000) 
        gpu_work_queue = ctx.Queue()
        actor_response_queues = [ctx.Queue() for _ in range(num_episodes)]
        num_dispatchers = getattr(self.args, 'num_dispatchers', 1) 
        gpu_result_queues = [ctx.Queue() for _ in range(num_dispatchers)]
        game_n = getattr(self.game, 'n', 8)
        
        target_model_filename = 'best.pth.tar'
        if self.args.load_model:
            self.nnet.load_checkpoint(self.args.load_folder_file[0], self.args.load_folder_file[1])
        else:
            if not os.path.exists(self.args.checkpoint): os.makedirs(self.args.checkpoint)
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=target_model_filename)
            
        from GpuWorker import start_gpu_worker_process
        game_params = {'n': game_n} 
        network_params = {'checkpoint': self.args.checkpoint, 'filename': target_model_filename, 'model_path': 'YES'}
        
        gpu_worker_proc = start_gpu_worker_process(gpu_work_queue, gpu_result_queues, self.args, "AmazonsGame", game_params, "NNetWrapper", network_params)
        self.process_manager.add_process(gpu_worker_proc)

        from Dispatcher import start_dispatcher_process
        dispatcher_procs = []
        for i in range(num_dispatchers):
            dp = start_dispatcher_process(i, shared_request_queue, actor_response_queues, gpu_work_queue, gpu_result_queues, self.args, None)
            dispatcher_procs.append(dp); self.process_manager.add_process(dp)
            
        import gc; results = []; pool = None; p1_wins = 0; p2_wins = 0; draws = 0; eps_count = 0
        
        try:
            pool = ctx.Pool(processes=min(self.args.num_actors, num_episodes), initializer=initialize_worker,
                initargs=(shared_request_queue, actor_response_queues, None, num_episodes, game_n))
            job_args = [(i, "AmazonsGame", game_params, self.args, network_params) for i in range(num_episodes)]
            job_results_iter = pool.imap(single_arg_wrapper_for_imap, [(a,) for a in job_args])
            
            for result_tuple in tqdm(job_results_iter, total=len(job_args), desc="Self Play"):
                train_data, winner = result_tuple
                results.extend(train_data)
                eps_count += 1
                if winner == 1: p1_wins += 1
                elif winner == -1: p2_wins += 1
                else: draws += 1
                if eps_count % 32 == 0:
                    log.info(f"\n[Stats] P1: {p1_wins} | P2: {p2_wins} | D: {draws}")
        except Exception as e:
            log.error(f"Exec Fail: {e}"); raise
        finally:
            if pool: pool.terminate(); pool.join()
            if gpu_worker_proc.is_alive(): gpu_worker_proc.terminate(); gpu_worker_proc.join()
            for p in dispatcher_procs: 
                if p.is_alive(): p.terminate(); p.join()
            gc.collect()
        return results

    def run_selfplay(self):
        try:
            import amazons_ops
            teacher = amazons_ops.ReplayBuffer(2000000) 
            BRIDGE_AVAILABLE = True
        except ImportError: BRIDGE_AVAILABLE = False
            
        for i in range(1, self.args.numIters + 1):
            log.info(f'Self Play Iteration {i} ...')
            iter_examples = deque([], maxlen=self.args.maxlenOfQueue)
            new_examples = self.executeParallelEpisodes(self.args.numEps)
            iter_examples.extend(new_examples)
            self.save_unique_examples(i - 1, list(iter_examples))
            self.trainExamplesHistory.append(iter_examples)
            
            if BRIDGE_AVAILABLE and hasattr(teacher, 'add_sample'):
                log.info("🌉 Bridging data to C++ Buffer...")
                for ex in new_examples:
                    # [修正] 数据桥接逻辑
                    # ex[0]: canonical board (已旋转180度，当前玩家=1，对手=-1)
                    # ex[1]: player (1 或 -1)
                    # ex[2]: policy tuples (src, dst, arr, prob)
                    # ex[3]: reward (1=当前玩家赢, -1=当前玩家输)
                    
                    canonical_board = ex[0].astype(np.int32)
                    player = ex[1]
                    
                    # C++ 端视角逻辑：
                    # - player_turn = 0 (Black): 不旋转，coor[0]=Black, coor[1]=White
                    # - player_turn = 1 (White): 旋转180度，coor[0]=Black, coor[1]=White
                    # 
                    # Python 端 canonical board 已经旋转180度，当前玩家=1，对手=-1
                    # 需要转换为 C++ 格式：
                    # - 反转数值：当前玩家(1) -> -1, 对手(-1) -> 1
                    # - 这样 C++ 端存储时，coor[0] (Black) = 原对手位置, coor[1] (White) = 原当前玩家位置
                    # - 传入 player_turn = 0，C++ 不旋转，直接使用
                    # - 当 C++ 读取时，如果 player_turn = 1，会旋转回来
                    
                    board_for_cpp = -canonical_board  # 反转数值
                    cpp_player = 0  # 总是传入 0，让 C++ 不旋转（因为已经旋转过了）
                    
                    # Winner: C++ 端 winner=0 表示 Black 赢，winner=1 表示 White 赢
                    # Python 端 ex[3] > 0 表示当前玩家赢
                    # 由于我们反转了数值，当前玩家在 C++ 端是 White (coor[1])
                    # 所以：如果当前玩家赢，winner=1 (White wins)
                    cpp_winner = 1 if ex[3] > 0 else 0
                    
                    tuples = ex[2]
                    if not tuples: continue
                    srcs = np.array([t[0] for t in tuples], dtype=np.int32)
                    dsts = np.array([t[1] for t in tuples], dtype=np.int32)
                    arrs = np.array([t[2] for t in tuples], dtype=np.int32)
                    probs = np.array([t[3] for t in tuples], dtype=np.float32)
                    
                    teacher.add_sample(board_for_cpp, cpp_player, cpp_winner, srcs, dsts, arrs, probs)
                teacher.save_data(os.path.join(self.args.checkpoint, "teacher_data_selfplay.bin"))

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                self.trainExamplesHistory.pop(0)
            
            self.run_training()
            self.run_arena()

    def run_training(self):
        log.info(">>> Starting Training Loop (Bridged C++ Data) <<<")
        try:
            import amazons_ops
            DATA_FILE = os.path.join(self.args.checkpoint, "teacher_data_selfplay.bin")
            JOINT_FILE = os.path.join(self.args.checkpoint, "teacher_data_joint.bin")
            teacher = amazons_ops.ReplayBuffer(2000000)
            
            if os.path.exists(DATA_FILE):
                log.info(f"Loading SelfPlay Data: {DATA_FILE}")
                teacher.load_data(DATA_FILE)
            elif os.path.exists(JOINT_FILE):
                log.info(f"Loading Joint Data: {JOINT_FILE}")
                teacher.load_data(JOINT_FILE)
            else:
                log.warning("No C++ Data found. Skipping Training.")
                return

            BATCH_SIZE = 256
            MAX_ACTIONS_PER_PASS = 16384
            TRAIN_STEPS = 200
            LR = 2e-4
            
            optimizer = optim.AdamW(self.nnet.nnet.parameters(), lr=LR, weight_decay=1e-4)
            self.nnet.nnet.train()
            device = self.nnet.device
            
            ploss_meter = AverageMeter()
            vloss_meter = AverageMeter()
            
            for step in range(TRAIN_STEPS):
                try:
                    batch_data = teacher.get_batch(BATCH_SIZE)
                except Exception as e:
                    log.error(f"Get Batch Fail: {e}"); break
                
                unique_boards = torch.from_numpy(batch_data[0]).to(device)
                b_idxs_all = torch.from_numpy(batch_data[1]).long().to(device)
                srcs_all   = torch.from_numpy(batch_data[2]).long().to(device)
                dsts_all   = torch.from_numpy(batch_data[3]).long().to(device)
                arrs_all   = torch.from_numpy(batch_data[4]).long().to(device)
                probs_all  = torch.from_numpy(batch_data[5]).float().to(device)
                vs_all     = torch.from_numpy(batch_data[6]).float().to(device)
                
                total_actions = srcs_all.size(0)
                optimizer.zero_grad()
                global_features = self.nnet.nnet.extract_features(unique_boards)
                
                perm = torch.randperm(total_actions, device=device)
                start_idx = 0
                total_ce = 0; total_v_loss = 0
                
                while start_idx < total_actions:
                    end_idx = min(start_idx + MAX_ACTIONS_PER_PASS, total_actions)
                    slice_indices = perm[start_idx:end_idx]
                    retain_graph = (end_idx < total_actions)
                    
                    b_idxs = b_idxs_all[slice_indices]
                    srcs   = srcs_all[slice_indices]
                    dsts   = dsts_all[slice_indices]
                    arrs   = arrs_all[slice_indices]
                    probs  = probs_all[slice_indices]
                    vs     = vs_all[b_idxs]
                    
                    log_p_move, log_p_arrow, out_v = self.nnet.nnet.forward_heads(
                        global_features, srcs, dsts, b_idxs
                    )
                    
                    ce_sum, _, _, _ = calculate_metrics(log_p_move, log_p_arrow, srcs, dsts, arrs, probs)
                    raw_mse = (out_v.view(-1) - vs) ** 2
                    weighted_mse = raw_mse * probs
                    v_loss_sum = weighted_mse.sum()
                    
                    loss_micro = (ce_sum + v_loss_sum) / BATCH_SIZE
                    
                    if torch.isnan(loss_micro):
                        log.error("NaN in Coach Training!"); break
                    
                    loss_micro.backward(retain_graph=retain_graph)
                    total_ce += ce_sum.item(); total_v_loss += v_loss_sum.item()
                    start_idx = end_idx
                
                torch.nn.utils.clip_grad_norm_(self.nnet.nnet.parameters(), 1.0)
                optimizer.step()
                ploss_meter.update(total_ce / BATCH_SIZE, BATCH_SIZE)
                vloss_meter.update(total_v_loss / BATCH_SIZE, BATCH_SIZE)
                if step % 50 == 0:
                    log.info(f"Train Step {step} | L_pi: {ploss_meter.avg:.4f} L_v: {vloss_meter.avg:.4f}")
            
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
        except Exception as e:
            log.error(f"Training Loop Failed: {e}")
            import traceback; traceback.print_exc()

    def run_arena(self):
        log.info(">>> Arena: NEW vs BEST <<<")
        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp_new_model.pth.tar')
        
        best_path = os.path.join(self.args.checkpoint, 'best.pth.tar')
        if not os.path.exists(best_path):
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            return

        MAX_ARENA_ACTORS = 32  
        ARENA_DISPATCHERS = 1
        num_games = self.args.arenaCompare
        actual_workers = min(MAX_ARENA_ACTORS, num_games)
        
        ctx = mp.get_context('spawn')
        new_req_q = ctx.Queue(); best_req_q = ctx.Queue()
        new_gpu_work_q = ctx.Queue(); best_gpu_work_q = ctx.Queue()
        new_gpu_res_qs = [ctx.Queue() for _ in range(ARENA_DISPATCHERS)]
        best_gpu_res_qs = [ctx.Queue() for _ in range(ARENA_DISPATCHERS)]
        new_resp_qs = [ctx.Queue() for _ in range(num_games)] 
        best_resp_qs = [ctx.Queue() for _ in range(num_games)]
        
        cleanup_procs = []
        pool = None
        
        try:
            from GpuWorker import start_gpu_worker_process
            from Dispatcher import start_dispatcher_process
            
            game_n = getattr(self.game, 'n', 8); game_params = {'n': game_n}
            arena_args = copy.copy(self.args)
            arena_args.inference_batch_size = 32
            
            p1 = start_gpu_worker_process(new_gpu_work_q, new_gpu_res_qs, arena_args, "AmazonsGame", game_params, "NNetWrapper", 
                {'checkpoint': self.args.checkpoint, 'filename': 'temp_new_model.pth.tar', 'model_path': 'YES'})
            cleanup_procs.append(p1)
            
            p2 = start_gpu_worker_process(best_gpu_work_q, best_gpu_res_qs, arena_args, "AmazonsGame", game_params, "NNetWrapper", 
                {'checkpoint': self.args.checkpoint, 'filename': 'best.pth.tar', 'model_path': 'YES'})
            cleanup_procs.append(p2)
            
            for i in range(ARENA_DISPATCHERS):
                d1 = start_dispatcher_process(i, new_req_q, new_resp_qs, new_gpu_work_q, new_gpu_res_qs, arena_args, None)
                d2 = start_dispatcher_process(i, best_req_q, best_resp_qs, best_gpu_work_q, best_gpu_res_qs, arena_args, None)
                cleanup_procs.extend([d1, d2])
            
            tasks = []
            half = num_games // 2
            for i in range(half): tasks.append((i, True, "AmazonsGame", game_params, self.args))
            for i in range(half, num_games): tasks.append((i, False, "AmazonsGame", game_params, self.args))
            
            pool = ctx.Pool(processes=actual_workers, initializer=initialize_arena_worker, 
                            initargs=(new_req_q, new_resp_qs, best_req_q, best_resp_qs))
            
            results_iter = pool.imap_unordered(wrapper_arena_parallel, [(t,) for t in tasks])
            nwins, bwins, draws = 0, 0, 0
            
            for res_tuple in tqdm(results_iter, total=num_games, desc="Arena"):
                res, _, _, _ = res_tuple
                if res == 1: nwins += 1
                elif res == -1: bwins += 1
                else: draws += 1
            
            log.info(f"Arena Result: NEW={nwins}, BEST={bwins}, DRAW={draws}")
            
            if nwins + bwins > 0 and float(nwins)/(nwins+bwins) > self.args.updateThreshold:
                log.info("ACCEPTING NEW MODEL")
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            else:
                log.info("REJECTING NEW MODEL")
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                
        except Exception as e:
            log.error(f"Arena Failed: {e}")
            import traceback; traceback.print_exc()
        finally:
            if pool: pool.terminate(); pool.join()
            for p in cleanup_procs: 
                if p.is_alive(): p.terminate(); p.join()

    def save_unique_examples(self, iteration, examples_data):
        from datetime import datetime; folder = self.args.checkpoint
        if not os.path.exists(folder): os.makedirs(folder)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(folder, f"examples_{timestamp}_iter{iteration}.examples")
        with open(filename, "wb+") as f: Pickler(f).dump(examples_data)