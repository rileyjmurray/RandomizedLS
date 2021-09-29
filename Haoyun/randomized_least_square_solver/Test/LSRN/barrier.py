"""	
Lisandro Dalcin:

Below, an scalable point-to-point based implementation of barrier() 
with the sleep() trick you need. A standard implementation would just 
merge send() and recv() on a single sendrecv() call. Just may need to 
tweak the sleep interval, and perhaps use a different tag value to 
avoid previous on-going communication.
"""

import time


def barrier(comm, tag=0, sleep=0.01): 
    size = comm.Get_size() 
    if size == 1: 
        return 
    rank = comm.Get_rank() 
    mask = 1 
    while mask < size: 
        dst = (rank + mask) % size 
        src = (rank - mask + size) % size 
        req = comm.isend(None, dst, tag) 
        while not comm.Iprobe(src, tag): 
            time.sleep(sleep) 
        comm.recv(None, src, tag) 
        req.Wait() 
        mask <<= 1 
