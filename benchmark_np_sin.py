import numpy as np
import scipy 
import time
from threading import Thread

from common.util.utils import timefunc

finished = False

DIM = 1000

def thread_worker():
    x = np.random.rand(33000) 
    while not finished:
        np.sin(x)
        np.cos(x)
        time.sleep(0.00001)

def start_worker(num):
    threads = []
    for i in xrange(num):
        threads.append(Thread(target=thread_worker))
        threads[i].start()
 
@timefunc
def do_sin(x):
    np.sin(x)
    np.cos(x)
    np.sin(x)

def main():
    start_worker(3)

    x = np.random.rand(DIM) 

    for i in xrange(20000):
        do_sin(x)
   
    finished = True

if __name__ == "__main__":
    main()
