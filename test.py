import multiprocessing
from multiprocessing.connection import wait
import time

def worker(pipe, message, delay):
    time.sleep(delay)
    pipe.send(message)
    pipe.close()

def parent():
    pipe1, child_pipe1 = multiprocessing.Pipe()
    pipe2, child_pipe2 = multiprocessing.Pipe()

    # 启动子进程
    p1 = multiprocessing.Process(target=worker, args=(child_pipe1, "Message from pipe1", 2))
    p2 = multiprocessing.Process(target=worker, args=(child_pipe2, "Message from pipe2", 20))
    p1.start()
    p2.start()

    pipes = [pipe1, pipe2]
    while pipes:
        ready_pipes = wait(pipes)  # 等待直到至少一个管道有数据
        print(ready_pipes)
        for pipe in ready_pipes:
            message = pipe.recv()  # 安全读取数据
            print(f"Received: {message}")
            pipes.remove(pipe)  # 移除已完成的管道

    # 确保子进程结束
    p1.join()
    p2.join()
    
class Env:
    def __init__(self, args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get_info(self):
        print(self.kwargs)

if __name__ == '__main__':
    env = Env({}, a=1, b=2)
    print(env.get_info())
    # parent()
