import time
import torch

print("convert pin memory time......................")
a = torch.rand((20*1024, 20*1024))
start = time.time()
a = a.pin_memory()
print(time.time() - start)
del a

print("convert cpu half time......................")
b = torch.rand((20*1024, 20*1024))
start = time.time()
b = b.half()
print(time.time() - start)
del b

print("copy cpu half time......................")
c = torch.rand((20*1024, 20*1024))
start = time.time()
c_e = torch.empty((20*1024, 20*1024), dtype=torch.half)
c_e.copy_(c)
print(time.time() - start)
del c
del c_e

print("copy cpu pin_memory time......................")
d = torch.rand((20*1024, 20*1024))
start = time.time()
d_e = torch.empty((20*1024, 20*1024), pin_memory=True)
d_e.copy_(d)
print(time.time() - start)
del d
del d_e

print("copy cpu pin_memory and half time......................")
e = torch.rand((20*1024, 20*1024), dtype=torch.float)
start = time.time()
e_e = torch.empty((20*1024, 20*1024), dtype=torch.half, pin_memory=True)
e_e.copy_(e)
print(time.time() - start)
del e
del e_e

print("copy cpu pin_memory to cuda time......................")
f = torch.rand((20*1024, 20*1024), dtype=torch.half, pin_memory=True)
torch.cuda.synchronize()
start = time.time()
f_cuda = torch.empty((20*1024, 20*1024), dtype=torch.half, device="cuda")
f_cuda.copy_(f)
torch.cuda.synchronize()
print(time.time() - start)
del f
del f_cuda

print("copy cuda to cpu pin_memory time......................")
g = torch.rand((20*1024, 20*1024), dtype=torch.half, device="cuda")
torch.cuda.synchronize()
start = time.time()
g_cpu = torch.empty((20*1024, 20*1024), dtype=torch.half, pin_memory=True)
g_cpu.copy_(g)
torch.cuda.synchronize()
print(time.time() - start)
del g
del g_cpu