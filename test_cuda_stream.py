import time
import torch

a = torch.rand((4096, 4096), pin_memory=True)
b = torch.rand((256, 256), device="cuda")
c = torch.rand((256, 256), device="cuda")

# s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

torch.cuda.synchronize()
start = time.time()

# with torch.cuda.stream(s1):
d = b * c

with torch.cuda.stream(s2):
    a_cuda = torch.empty((1024, 1024), device="cuda")
    a_cuda.copy_(a, non_blocking=True)

torch.cuda.synchronize()
print("asyn", time.time() - start)

torch.cuda.synchronize()
start = time.time()
e = b * c
a_cuda = torch.empty((1024, 1024), device="cuda")
a_cuda.copy_(a, non_blocking=True)
torch.cuda.synchronize()
print("syn", time.time() - start)