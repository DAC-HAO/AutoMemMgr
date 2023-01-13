import time
import torch

a = torch.rand((1024, 1024))

start_time = time.time()
pin_a = a.pin_memory()
print("to pin time", time.time() - start_time)

torch.cuda.synchronize()
start_time = time.time()
cuda_pin_a = pin_a.to("cuda")
torch.cuda.synchronize()
print("pin to cuda time", time.time() - start_time)

torch.cuda.synchronize()
start_time = time.time()
cuda_a = a.to("cuda")
torch.cuda.synchronize()
print("cpu to cuda time", time.time() - start_time)



# a = torch.rand((16*1024, 16*1024), device="cuda:0")
# b = torch.rand((16*1024, 16*1024), device="cuda:0")
# torch.cuda.synchronize()
# start = time.time()
# c = a * b
# torch.cuda.synchronize()
# print(time.time() - start)
#
#
# cpu_data = torch.rand((4096, 4096), pin_memory=True)
# torch.cuda.synchronize()
# start = time.time()
# cuda_data = torch.empty((4096, 4096), device="cuda:0")
# cuda_data.copy_(cpu_data, non_blocking=True)
# torch.cuda.synchronize()
# print(time.time()-start)