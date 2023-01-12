import time
import torch

a = torch.rand((2, 2))

start_time = time.time()
pin_a = a.pin_memory()
print("to pin time", time.time() - start_time)

torch.cuda.synchronize()
start_time = time.time()
cuda_pin_a = pin_a.to("cuda")
torch.cuda.synchronize()
print("pin to cuda time", time.time() - start_time)

pin_a[0] = 0
print(a, pin_a)

torch.cuda.synchronize()
start_time = time.time()
cuda_a = a.to("cuda")
torch.cuda.synchronize()
print("cpu to cuda time", time.time() - start_time)

