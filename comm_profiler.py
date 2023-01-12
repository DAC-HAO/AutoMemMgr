import time
import torch

a = torch.rand((1024, 1024))

start_time = time.time()
pin_a = a.pin_memory()
print("to pin time", time.time() - start_time)

torch.cuda.synchronize()
start_time = time.time()
cuda_pin_a = pin_a.cuda()
torch.cuda.synchronize()
print("pin to cuda time", time.time() - start_time)

torch.cuda.synchronize()
start_time = time.time()
cuda_a = a.cuda()
torch.cuda.synchronize()
print("cpu to cuda time", time.time() - start_time)

