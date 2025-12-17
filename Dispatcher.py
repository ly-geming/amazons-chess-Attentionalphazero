import torch
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import logging
import queue
import sys
import os
import psutil
import traceback

log = logging.getLogger(__name__)

def set_high_priority():
    try:
        p = psutil.Process(os.getpid())
        if os.name == 'nt': p.nice(psutil.HIGH_PRIORITY_CLASS)
        else: p.nice(-10)
        print(f"Dispatcher Process {os.getpid()} set to HIGH Priority.", flush=True)
    except Exception as e: print(f"Failed to set priority: {e}", flush=True)

class Dispatcher:
    def __init__(self, dispatcher_id, shared_request_queue, actor_response_queues_list, 
                 gpu_work_queue, gpu_result_queues_list, args_dict, shm_info=None):
        self.dispatcher_id = dispatcher_id
        self.shared_request_queue = shared_request_queue
        self.actor_response_queues_list = actor_response_queues_list
        self.gpu_work_queue = gpu_work_queue
        self.gpu_result_queues_list = gpu_result_queues_list

        class ArgsObj:
            def __init__(self, d):
                for k, v in d.items(): setattr(self, k, v)
        args = ArgsObj(args_dict)
        self.args = args
        
        self.batch_size = getattr(args, 'inference_batch_size', getattr(args, 'batch_size', 64))
        self.initial_timeout = 0.000001
        self.continuous_timeout = 0.001
        
        # [SHM Connection]
        self.shm = None
        self.shared_board_buffer = None
        if shm_info:
            try:
                self.shm = shared_memory.SharedMemory(name=shm_info['name'])
                self.shared_board_buffer = np.ndarray(
                    (shm_info['num_actors'], shm_info['shape'][0], shm_info['shape'][1]),
                    dtype=np.int32,
                    buffer=self.shm.buf
                )
                log.info(f"Dispatcher {dispatcher_id} connected to SHM.")
            except Exception as e:
                log.error(f"Dispatcher SHM Connection Fail: {e}")

    def run(self):
        set_high_priority()
        print(f"Dispatcher {self.dispatcher_id} running. BatchSize={self.batch_size}", flush=True)

        while True:
            actor_ids_shm = []      
            actor_ids_fallback = []  
            boards_fallback = []     
            final_actor_id_order = []

            start_time = time.time()
            
            # --- Collection Phase ---
            try:
                item = self.shared_request_queue.get(timeout=self.initial_timeout)
                if isinstance(item, int): 
                    actor_ids_shm.append(item)
                    final_actor_id_order.append(item)
                else: 
                    actor_ids_fallback.append(item[0])
                    boards_fallback.append(item[1])
                    final_actor_id_order.append(item[0])
            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"Dispatcher Queue Error: {e}"); continue

            collection_time_limit = 0.05
            while len(final_actor_id_order) < self.batch_size and (time.time() - start_time) < collection_time_limit:
                try:
                    item = self.shared_request_queue.get(timeout=self.continuous_timeout)
                    if isinstance(item, int):
                        actor_ids_shm.append(item)
                        final_actor_id_order.append(item)
                    else:
                        actor_ids_fallback.append(item[0])
                        boards_fallback.append(item[1])
                        final_actor_id_order.append(item[0])
                except queue.Empty: break 
                except Exception: break

            # --- Processing Phase ---
            if final_actor_id_order:
                try:
                    batch_parts = []
                    
                    if actor_ids_shm:
                        if self.shared_board_buffer is not None:
                            batch_parts.append(self.shared_board_buffer[actor_ids_shm].astype(np.float64))
                        else:
                            log.critical("Received SHM ID but SHM buffer is None! Dropping requests.")
                            sys.exit(1)
                            
                    if boards_fallback:
                        batch_parts.append(np.array(boards_fallback, dtype=np.float64))
                        
                    if not batch_parts: continue
                        
                    if len(batch_parts) == 1: boards_numpy = batch_parts[0]
                    else: boards_numpy = np.concatenate(batch_parts, axis=0)
                        
                    current_batch_actor_ids = actor_ids_shm + actor_ids_fallback

                    # [IPC Send to GPU]
                    self.gpu_work_queue.put((boards_numpy, self.dispatcher_id), timeout=2.0)

                    # [IPC Wait Result]
                    try:
                        # [Modified] 接收结构: ((moves, arrows), vs)
                        (batch_moves, batch_arrows), batch_vs = self.gpu_result_queues_list[self.dispatcher_id].get(timeout=10.0)
                    except queue.Empty:
                        log.critical(f"Dispatcher {self.dispatcher_id}: GPU Deadlock (10s timeout)!")
                        sys.exit(1)

                    # [IPC Dispatch Response]
                    for i, actor_id in enumerate(current_batch_actor_ids):
                        if 0 <= actor_id < len(self.actor_response_queues_list):
                            # [Modified] 组装单个 Actor 的响应
                            # p_move: (4096,)
                            # p_arrow: (64, 64)
                            single_pi = (batch_moves[i], batch_arrows[i])
                            single_v = batch_vs[i]
                            
                            self.actor_response_queues_list[actor_id].put((single_pi, single_v), timeout=1.0)
                        else:
                            log.error(f"Invalid Actor ID: {actor_id}")

                except Exception as e:
                    log.error(f"Dispatcher Critical Error: {e}")
                    traceback.print_exc()
                    sys.exit(1)

def dispatcher_process_main(dispatcher_id, shared_request_queue, actor_response_queues_list, gpu_work_queue, gpu_result_queues_list, args_dict, shm_info=None):
    try:
        dispatcher = Dispatcher(dispatcher_id, shared_request_queue, actor_response_queues_list, gpu_work_queue, gpu_result_queues_list, args_dict, shm_info)
        dispatcher.run()
    except KeyboardInterrupt: pass
    except Exception as e: print(f"Dispatcher crashed: {e}")

def start_dispatcher_process(dispatcher_id, shared_request_queue, actor_response_queues_list, gpu_work_queue, gpu_result_queues_list, args_obj, shm_info=None):
    args_dict = {}
    for attr in dir(args_obj):
        if not attr.startswith('_') and not callable(getattr(args_obj, attr)):
            try:
                val = getattr(args_obj, attr)
                if isinstance(val, (str, int, float, bool, list, dict, tuple)) or val is None: args_dict[attr] = val
            except: pass
    p = mp.Process(target=dispatcher_process_main, args=(dispatcher_id, shared_request_queue, actor_response_queues_list, gpu_work_queue, gpu_result_queues_list, args_dict, shm_info))
    p.start(); return p