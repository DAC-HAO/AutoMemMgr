from typing import List, Dict
import torch
import torch.fx
from torch.fx import GraphModule
import torch.optim as optim

from torch.utils._pytree import tree_map

from colossalai.fx import ColoTracer, is_compatible_with_meta
from colossalai.fx.passes.meta_info_prop import MetaInfoProp

from colossalai.amp.naive_amp import FP16Optimizer
from colossalai.amp.naive_amp.grad_scaler import DynamicGradScaler

from strategies_constructor import OffloadStrategiesConstructor
from solver import Solver
from runtime import runtime_offload_apply_pass
from basic_offload_module import BasicOffloadModule

if is_compatible_with_meta():
    from colossalai.fx.profiler import MetaTensor

def memory_optimization(model: torch.nn.Module, inps: Dict[str, torch.Tensor]):
    model.cpu()
    tracer = ColoTracer()
    wrap_fn = lambda x: MetaTensor(x, fake_device=torch.device("cpu")) if isinstance(x, torch.Tensor) else x
    meta_args = tree_map(wrap_fn, inps)
    graph = tracer.trace(model, meta_args=meta_args)
    gm = GraphModule(model, graph, model.__class__.__name__)

    interp = MetaInfoProp(gm)
    interp.propagate(*meta_args.values())

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-09)
    optimizer = FP16Optimizer(optimizer, DynamicGradScaler())
    offload_strategies_constructor = OffloadStrategiesConstructor(graph, optimizer)
    offload_strategies_constructor.build_strategies_and_cost()

    solver = Solver(gm.graph, offload_strategies_constructor)
    solver._call_solver_greedy_v1()

    gm = runtime_offload_apply_pass(gm)
    gm.recompile()
    optimized_model = BasicOffloadModule(gm)
    return optimized_model, optimizer
