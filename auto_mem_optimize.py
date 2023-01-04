from typing import List, Dict
import torch
import torch.fx
from torch.fx import GraphModule

from torch.utils._pytree import tree_map

from colossalai.fx import ColoTracer, is_compatible_with_meta

if is_compatible_with_meta():
    from colossalai.fx.profiler import MetaTensor

def memory_optimization(model: torch.nn.Module, inps: Dict[str, torch.Tensor]) -> GraphModule:
    model.cpu()
    tracer = ColoTracer()
    wrap_fn = lambda x: MetaTensor(x, fake_device=torch.device("cpu")) if isinstance(x, torch.Tensor) else x
    graph = tracer.trace(model, meta_args=tree_map(wrap_fn, inps))
    gm = GraphModule(model, graph, model.__class__.__name__)

    # TODO: checkpoint and offload strategy solver

    gm.recompile()
    return gm