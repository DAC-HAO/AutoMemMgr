
import torch
import torch.nn as nn
from torch.fx import GraphModule
from torch.utils._pytree import tree_map
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.fx import ColoTracer
from colossalai.fx.passes.meta_info_prop import MetaInfoProp


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 4, bias=False)
        self.fc2 = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out

with ColoInitContext(device=torch.device("cuda")):
    model = MyModel()
data_dict = {"x" : torch.rand((1, 4), device="cuda")}

tracer = ColoTracer()
wrap_fn = lambda x: x.to("meta") if isinstance(x, torch.Tensor) else x
meta_args = tree_map(wrap_fn, data_dict)
graph = tracer.trace(model, meta_args=meta_args)
gm = GraphModule(model, graph, model.__class__.__name__)
gm.recompile()

interp = MetaInfoProp(gm)
interp.propagate(*meta_args.values())

