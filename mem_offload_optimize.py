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
from solver import Solver, AsynGreedySolver
from runtime import runtime_offload_apply_pass
from basic_offload_module import BasicOffloadModule, AMPOptimizer


def memory_optimization(model: torch.nn.Module, inps: Dict[str, torch.Tensor], memory_budget: float=-1.0):
    model.cpu()
    tracer = ColoTracer()
    assert is_compatible_with_meta()
    # wrap_fn = lambda x: MetaTensor(x, fake_device=torch.device("cpu")) if isinstance(x, torch.Tensor) else x
    wrap_fn = lambda x: x.to("meta") if isinstance(x, torch.Tensor) else x
    meta_args = tree_map(wrap_fn, inps)
    graph = tracer.trace(model, meta_args=meta_args)
    gm = GraphModule(model, graph, model.__class__.__name__)

    interp = MetaInfoProp(gm)
    interp.propagate(*meta_args.values())

    # optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-09)
    # optimizer = FP16Optimizer(optimizer, DynamicGradScaler())
    # optimizer = AMPOptimizer(optimizer, DynamicGradScaler())

    offload_strategies_constructor = OffloadStrategiesConstructor(graph)
    offload_strategies_constructor.build_strategies_and_cost()

    # solver = Solver(gm.graph, offload_strategies_constructor, memory_budget)
    # solver._call_solver_greedy_v1()
    # solver._call_solver_l2l()

    solver = AsynGreedySolver(gm.graph, memory_budget)
    solver._call_solver_greedy()

    # print offload node
    for node in graph.nodes:
        if node.node_info.offload_param_flag:
            print(node.op, node.name, node.node_info.node_to_prefetch)

    gm = runtime_offload_apply_pass(gm)
    gm.recompile()
    optimized_model = BasicOffloadModule(gm)
    return optimized_model