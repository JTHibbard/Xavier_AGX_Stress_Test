import numpy as np
from timeit import default_timer as timer
from numba import vectorize
from multiprocessing import Process, Lock
import time


@vectorize(['float64(float64, float64)'], target='cuda')
def pow_gpu(a, b):
    c = a ** b
    for j in range(25):
        c = c ** (1 / b)
        c = c ** b
    return c


def gpu_loop(loc):
    loc.acquire()
    try:
        print("GPU process has started")
    finally:
        loc.release()
    vec_size = 105
    count = 0

    a = b = np.array(np.random.rand(vec_size, vec_size, vec_size, vec_size), dtype=np.float64)
    c = np.zeros(vec_size, dtype=np.float64)

    while True:
        start = timer()
        c = pow_gpu(a, b)
        print("GPU is still running, the load loop took in {0} seconds".format(timer() - start))


def pow_cpu(a, b, c):
    for j in range(a.size):
        c[j] = a[j] ** b[j]


def cpu_loop(loc, process_num):
    loc.acquire()
    try:
        print("Process Number {0} has started".format(process_num))
    finally:
        loc.release()
    vec_size = 10000000

    a = b = np.array(np.random.sample(vec_size), dtype=np.float64)
    c = np.zeros(vec_size, dtype=np.float64)
    start = timer()

    while True:
        pow_cpu(a, b, c)
        loc.acquire()
        try:
            print("Process {0} is still running, the load loop took {1} seconds".format(process_num, timer() - start))
        finally:
            loc.release()

        start = timer()


if __name__ == '__main__':
    cpu_cores = 8
    lock = Lock()
    all_processes = []
    for i in range(cpu_cores):
        process = Process(target=cpu_loop, args=(lock, i + 1))
        all_processes.append(process)
    for p in all_processes:
        p.start()
    gpu = Process(target=gpu_loop, args=(lock,))
    gpu.start()
    time.sleep(1)
    print("The loops in this program will run until you quit")
    time.sleep(1)
    while True:
        val = input('Type "q" at anytime to exit: ')
        if val.lower() == "q":
            break
    for p in all_processes:
        p.terminate()
    gpu.terminate()
