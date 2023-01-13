import time
import torch

# a = torch.rand((4096, 4096), pin_memory=False)
# b = torch.rand((16*1024, 16*1024), device="cuda")
# c = torch.rand((16*1024, 16*1024), device="cuda")
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

# # syn
# a = torch.rand((4096, 4096))
# torch.cuda.synchronize()
# start = time.time()
# e = b * c
# a_cuda = torch.empty((4096, 4096), device="cuda")
# a_cuda.copy_(a)
# torch.cuda.synchronize()
# print("syn", time.time() - start)



# for iii in range(10):
#     a = torch.rand((2*(iii+1) * 1024, 2*(iii+1) * 1024), device="cuda:0", dtype=torch.half)
#     b = torch.rand((2*(iii+1) * 1024, 2*(iii+1) * 1024), dtype=torch.half)
#     cuda_time = 0
#     cpu_time = 0
#     for jjj in range(20):
#         torch.cuda.synchronize()
#         start = time.time()
#         # half_a = a.half()
#         # float_a = a.to(dtype=torch.float32)
#         float_a = a.float()
#         torch.cuda.synchronize()
#         cuda_to_half = time.time() - start
#         cuda_time += cuda_to_half
#         # print("cuda to half", cuda_to_half)
#
#         start = time.time()
#         # half_b = b.half()
#         # float_b = b.to(dtype=torch.float32)
#         float_b = b.float()
#         cpu_to_half = time.time() - start
#         cpu_time += cpu_to_half
#         # print("cpu to half", cpu_to_half)
#
#         # del half_a
#         # del half_b
#         del float_a
#         del float_b
#
#     print(f'({2*(iii+1)}*1024,{2*(iii+1)}*1024)\t{cuda_time/20:.6f}\t{cpu_time/20:.6f}')
#     del a
#     del b



# a = torch.rand((20*1024, 20*1024), device="cuda:0", dtype=torch.float32)
# torch.cuda.synchronize()
# start = time.time()
# a = a.half()
# torch.cuda.synchronize()
# print(time.time() - start)



# for iii in range(10):
#     a = torch.rand((2*(iii+1) * 1024, 2*(iii+1) * 1024), device="cuda:0")
#     b = torch.rand((2*(iii+1) * 1024, 2*(iii+1) * 1024), device="cpu")
#     cuda_time = 0
#     cpu_time = 0
#     for jjj in range(20):
#         torch.cuda.synchronize()
#         start = time.time()
#         a_cpu = a.to(device="cpu")
#         torch.cuda.synchronize()
#         cuda_to_cpu = time.time() - start
#         cuda_time += cuda_to_cpu
#
#         torch.cuda.synchronize()
#         start = time.time()
#         b_cuda = b.to(device="cuda:0")
#         torch.cuda.synchronize()
#         cpu_to_cuda = time.time() - start
#         cpu_time += cpu_to_cuda
#
#         del a_cpu
#         del b_cuda
#
#     cuda_time/=20
#     cpu_time/=20
#     print(f'({2*(iii+1)}*1024,{2*(iii+1)}*1024)\t{cuda_time:.6f}\t{cpu_time:.6f}')
#     del a
#     del b


# 测试 H2D 性能
# a = torch.rand((20*1024, 20*1024), device="cpu", pin_memory=True)
# cuda_a = torch.empty((20*1024, 20*1024), device="cuda:0")
# torch.cuda.synchronize()
# start = time.time()
# # a = a.to(device="cuda:0")
# cuda_a.copy_(a)
# torch.cuda.synchronize()
# print(time.time() - start)


# 测试 D2H 性能
# a = torch.rand((20*1024, 20*1024), device="cuda:0")
# a_cpu = torch.empty((20*1024, 20*1024), device="cpu", pin_memory=False)
# torch.cuda.synchronize()
# start = time.time()
# # a = a.to(device="cpu")
# a_cpu.copy_(a)
# torch.cuda.synchronize()
# print(time.time() - start)


# # 测试 pin_memory 性能
# a = torch.rand((20*1024, 20*1024), device="cpu")
# start = time.time()
# a.pin_memory()
# print(time.time() - start)


# # 测试 cpu copy 性能
# a = torch.rand((20*1024, 20*1024), device="cpu")
# a_copy = torch.empty((20*1024, 20*1024), device="cpu")
# start = time.time()
# a_copy.copy_(a)
# print(time.time() - start)

