import multiprocessing as mp
import psutil
import time
import logging
import queue

log = logging.getLogger(__name__)

class ProcessManager:
    def __init__(self):
        self.processes = []
        self.queues = []
    
    def add_process(self, proc):
        self.processes.append(proc)
    
    def add_queue(self, queue):
        self.queues.append(queue)
    
    def terminate_all(self, timeout=10):
        """安全终止所有管理的进程"""
        # 先尝试优雅终止
        for proc in self.processes:
            if proc and proc.is_alive():
                try:
                    proc.terminate()
                except Exception as e:
                    log.warning(f"Error terminating process: {e}")
        
        # 等待进程结束
        for proc in self.processes:
            if proc:
                try:
                    proc.join(timeout=timeout)
                except Exception as e:
                    log.warning(f"Error joining process: {e}")
        
        # 如果仍有进程存活，强制终止
        for proc in self.processes:
            if proc and proc.is_alive():
                try:
                    proc.kill()
                    proc.join()
                except Exception as e:
                    log.warning(f"Error killing process: {e}")
        
        # 清理队列
        for queue_obj in self.queues:
            try:
                # 清空队列中的所有项目
                while True:
                    queue_obj.get_nowait()
            except queue.Empty:
                pass  # 队列为空，结束清理
        
        # 清空列表
        self.processes = []
        self.queues = []
    
    def cleanup_zombies(self):
        """清理僵尸进程"""
        try:
            parent = psutil.Process()
            children = parent.children(recursive=True)
            for child in children:
                if child.status() == 'zombie':
                    child.wait()
        except Exception as e:
            log.warning(f"Error cleaning zombies: {e}")