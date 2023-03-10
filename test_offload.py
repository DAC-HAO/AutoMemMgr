import random
import argparse
import time
import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from colossalai.fx.profiler import parameter_size

from mem_offload_optimize import memory_optimization

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # self.fc1 = nn.Linear(512, 1024, bias=False)
        # self.fc2 = nn.Linear(1024, 1024, bias=False)
        # self.fc3 = nn.Linear(1024, 2048, bias=False)
        # self.fc4 = nn.Linear(2048, 512, bias=False)
        # self.fc5 = nn.Linear(512, 512, bias=False)

        random.seed(100)
        self.fc_list = nn.ModuleList([])
        dim_list = [512, 1024, 2048]
        in_dim = dim_list[0]
        for iii in range(10):
            out_dim = dim_list[random.randint(0, 2)]
            self.fc_list.append(nn.Linear(in_dim, out_dim, bias=False))
            in_dim = out_dim


    def forward(self, x):
        # out = self.fc1(x)
        # out = self.fc2(out)
        # out = self.fc3(out)
        # out = self.fc4(out)
        # out = self.fc5(out)
        # return out

        for m in self.fc_list:
            x = m(x)
        return x


parser = argparse.ArgumentParser(description="offload testing")
# parser.add_argument("-m_name", type=str, default="simplenet",
#                     help="model name")
parser.add_argument("-mem_size", type=float, default=32, help="memory budget (MB)")
parser.add_argument('-is_syn', action='store_true', help='If true, offload is performed synchronously.')
args = parser.parse_args()

model = MyModel()
data_dict = {"x": torch.rand((1, 512))}

param_size = parameter_size(model)/1024**2
model = memory_optimization(model, data_dict, 1024*1024*args.mem_size, args.is_syn)
wrap_fn = lambda x: x.to("cuda") if isinstance(x, torch.Tensor) else x
data_dict = tree_map(wrap_fn, data_dict)

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
start_time = time.time()

# prof = torch.profiler.profile(
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/offload'),
#         record_shapes=True,
#         with_stack=True)
# prof.start()
# prof.step()
loss = torch.sum(model(**data_dict))
print(loss)
loss.backward()
# prof.step()
# prof.stop()

torch.cuda.synchronize()

exec_time = time.time() - start_time
runtime_peak_mem = torch.cuda.max_memory_allocated()/1024**2
print(
        f'|exec_time={exec_time:.3f} s | param_size={param_size:.3f} MB | runtime_peak_mem={runtime_peak_mem:.3f} MB|'
    )
