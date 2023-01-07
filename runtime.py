from dataclasses import dataclass
import torch
from torch.fx.node import Node
from colossalai.gemini.tensor_utils import alloc_storage, free_storage

from util import ModelParameters


class OffloadParameter(torch.autograd.Function):
    """
    A customized offload operation which forward is parameter release operation,
    backward is a parameter upload operation.

    Args:
        input_: input matrix.
        offload_node:.
    """

    @staticmethod
    def forward(ctx, input_, params_indices):
        # offload
        ctx.params_indices = params_indices
        for param_idx in params_indices:
            free_storage(ModelParameters.fp16_params[param_idx].data)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # prefetch
        for param_idx in ctx.params_indices:
            fp16_param = ModelParameters.fp16_params[param_idx]
            alloc_storage(fp16_param.data)
            fp16_param.data.copy_(ModelParameters.fp32_master_params[param_idx].data)
        return grad_output, None


def covert_spec_to_action(tensor, params_indices):
    '''
    Convert OffloadSpec into runtime action, implement offload operation target tensor.

    Argument:
        tensor(torch.Tensor): Tensor stored in each device, which could be different in different ranks.
    '''
    return OffloadParameter.apply(tensor, params_indices)


def runtime_offload_apply_pass(gm: torch.fx.GraphModule):
    """
    This pass is used to add the offload spec apply node to the origin graph.
    """
    mod_graph = gm.graph
    nodes = tuple(mod_graph.nodes)
    for node in nodes:
        if node.node_info.has_param:
            pass
        if node.node_info.offload_param_flag:
            param_indices = node.node_info.param_indices
            assert isinstance(param_indices, list)
            with mod_graph.inserting_after(node):
                offload_apply_node = mod_graph.create_node('call_function', covert_spec_to_action, args=(node, param_indices))
            user_list = list(node.users.keys())
            for user in user_list:
                if user == offload_apply_node:
                    continue
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



