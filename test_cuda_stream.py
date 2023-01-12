import time
import torch

# a = torch.rand((4096, 4096), pin_memory=True)
# b = torch.rand((256, 256), device="cuda")
# c = torch.rand((256, 256), device="cuda")
#
# comm_stream = torch.cuda.Stream()
#
# # asyn
# torch.cuda.synchronize()
# start = time.time()
# d = b * c
# with torch.cuda.stream(comm_stream):
#     a_cuda = torch.empty((4096, 4096), device="cuda")
#     a_cuda.copy_(a, non_blocking=True)
# torch.cuda.synchronize()
# print("asyn", time.time() - start)
#
# # syn
# torch.cuda.synchronize()
# start = time.time()
# e = b * c
# a_cuda = torch.empty((4096, 4096), device="cuda")
# a_cuda.copy_(a, non_blocking=True)
# torch.cuda.synchronize()
# print("syn", time.time() - start)

a = torch.rand((128, 128), device="cuda:0")
b = torch.rand((128, 128), device="cuda:0")
torch.cuda.synchronize()
start = time.time()
c = a * b
torch.cuda.synchronize()
print(time.time() - start)


cpu_data = torch.rand((4096, 4096), pin_memory=True)
torch.cuda.synchronize()
start = time.time()
cuda_data = torch.empty((4096, 4096), device="cuda:0")
cuda_data.copy_(cpu_data, non_blocking=True)
torch.cuda.synchronize()
print(time.time()-start)
