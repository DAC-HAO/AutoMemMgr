from dataclasses import dataclass
from typing import List, Dict
import torch
from torch.fx import Graph, Node

from offload_strategy import OffloadStrategiesVector


class ModelParameters:
    param_idx = 0
    fp16_params = []
    fp32_master_params = []

@dataclass
class NodeInfo:
    has_param: bool = False
    param_size: float = 0
    offload_param_flag: bool = False
    param_indices: List = None
    runtime_fwd_mem: float = 0
    runtime_bwd_mem: float = 0
    offload_strategies_vector: OffloadStrategiesVector = None

def move_to_cpu(node: Node):
    assert node.op in ['call_function', 'call_module']
    if node.op == 'call_function':
        pass

    elif node.op == 'call_module':
        pass


def move_to_cuda(node: Node):
    pass
