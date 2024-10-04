from timeit import default_timer as timer
import numpy as np
from numba import cuda,jit
import numba
import random


num=10000000
num2=1000000000 # *64 = 7.4GB

def withCPU(a):
    for i in range(num):
        a[i]+=random.random()

@jit
def withGPU(a):
    for i in range(num):
        a[i]+=random.random()

@cuda.jit
def cudaJit(a):
    row=cuda.grid(1)
    if row < a.shape[0]:
        a[row]+=1



a=np.ones(num2,dtype=np.float64)

print('number of iterations is ',num)

start=timer()# JIT
withGPU(a)
print('GPU TOOK ',(timer()-start),' SECONDS')

# start=timer()# CUDA JIT
# cudaJit(a)
# print('GPU TOOK ',(timer()-start),' SECONDS')

# start=timer()
# withCPU(a)
# print('CPU TOOK ',(timer()-start),' SECONDS')
