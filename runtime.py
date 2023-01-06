import torch
from torch.fx.node import Node
from colossalai.gemini.tensor_utils import alloc_storage, free_storage

class OffloadParameter(torch.autograd.Function):
    """
    A customized offload operation which forward is parameter release operation,
    backward is a parameter upload operation.

    Args:
        input_: input matrix.
        offload_node:.
    """

    @staticmethod
    def forward(ctx, input_, offload_node):
        # offload
        ctx.offload_node = offload_node
        for p in offload_node.fp16_params:
            free_storage(p.data)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # prefetch
        for idx, p in enumerate(ctx.offload_node.fp16_params):
            alloc_storage(p.data)
            p.data.copy_(ctx.offload_node.fp32_master_params[idx].data.half())
        return grad_output, None

def runtime_offload_apply_pass(gm: torch.fx.GraphModule):
    """
    This pass is used to add the offload spec apply node to the origin graph.
    """
    mod_graph = gm.graph
    for node in mod_graph.nodes:
        if node.meta['offload_param']:
            with mod_graph.inserting_after(node):
                offload_apply_node = mod_graph.create_node('call_function', OffloadParameter(), args=(node,))
            user_list = list(node.users.keys())
            for user in user_list:
                new_args = list(user.args)
                new_kwargs = dict(user.kwargs)
                # the origin node may be a positional argument or key word argument of user node
                if node in new_args:
                    # substitute the origin node with offload_apply_node
                    new_args[new_args.index(node)] = offload_apply_node
                    user.args = tuple(new_args)
                elif str(node) in new_kwargs:
                    # substitute the origin node with offload_apply_node
                    new_kwargs[str(node)] = offload_apply_node
                    user.kwargs = new_kwargs
    return gm



