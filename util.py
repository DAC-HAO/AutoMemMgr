import torch
from torch.fx import Graph, Node

def move_to_cpu(node: Node):
    assert node.op in ['call_function', 'call_module']
    if node.op == 'call_function':
        pass

    elif node.op == 'call_module':
        pass


def move_to_cuda(node: Node):
    pass