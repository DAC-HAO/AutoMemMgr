import torch
from torch.fx.graph import Graph
from torch.fx.node import Node
from colossalai.utils.cuda import get_current_device
from strategies_constructor import OffloadStrategiesConstructor

class Solver:

    def __init__(self,
                 graph: Graph,
                 strategies_constructor: OffloadStrategiesConstructor,
                 memory_budget: float = -1.0):
        self.graph = graph
        self.strategies_constructor = strategies_constructor
        self.memory_budget = memory_budget if memory_budget > 0 \
            else torch.cuda.get_device_properties(get_current_device()).total_memory

    def _call_solver_serialized_args(self):
        peak_mem = 0
        while peak_mem > self.memory_budget:
            pass