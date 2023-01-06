
import torch
import torch.nn as nn
from torch.fx import GraphModule
from torch.utils._pytree import tree_map
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.fx import ColoTracer

from mem_offload_optimize import memory_optimization

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 4, bias=False)
        self.fc2 = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


model = MyModel().half()
data_dict = {"x" : torch.rand((1, 4), dtype=torch.float16)}

tracer = ColoTracer()
wrap_fn = lambda x: x.to("meta") if isinstance(x, torch.Tensor) else x
meta_args = tree_map(wrap_fn, data_dict)
graph = tracer.trace(model, meta_args=meta_args)
gm = GraphModule(model, graph, model.__class__.__name__)
gm.recompile()
loss = torch.sum(gm(**data_dict))
loss.backward()
