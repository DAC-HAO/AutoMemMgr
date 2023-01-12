import time
import torch

# a = torch.rand((1024, 1024))
#
# start_time = time.time()
# pin_a = a.pin_memory()
# print("to pin time", time.time() - start_time)
#
# torch.cuda.synchronize()
# start_time = time.time()
# cuda_pin_a = pin_a.to("cuda")
# torch.cuda.synchronize()
# print("pin to cuda time", time.time() - start_time)
#
# torch.cuda.synchronize()
# start_time = time.time()
# cuda_a = a.to("cuda")
# torch.cuda.synchronize()
# print("cpu to cuda time", time.time() - start_time)


a = torch.rand((1024, 1024), pin_memory=True)
b = torch.rand((64, 64), device="cuda")
c = torch.rand((64, 64), device="cuda")

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

torch.cuda.synchronize()
start = time.time()

with torch.cuda.stream(s1):
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