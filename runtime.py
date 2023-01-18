from dataclasses import dataclass
import torch
from torch.fx.node import Node
from colossalai.gemini.tensor_utils import alloc_storage, free_storage

from util import ModelParameters, GlobalCudaInfo


class PreForwardUpload(torch.autograd.Function):
    """
    A customized upload operation which forward is parameter upload operation,
    backward is a parameter release operation.

    Args:
        input_: input matrix.
        params_indices:.
    """

    @staticmethod
    def forward(ctx, input_, params_indices):
        # upload
        ctx.params_indices = params_indices
        for param_idx in params_indices:
            fp16_param = ModelParameters.fp16_params[param_idx]
            if fp16_param.data.device.type == "cpu":
                fp16_param.data = fp16_param.data.to("cuda")
            else:
                print("PreForwardUpload", fp16_param.data.shape)
                alloc_storage(fp16_param.data)
                fp16_param.data.copy_(ModelParameters.fp32_master_params[param_idx].data)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # release
        # print(ctx.params_indices, grad_output.shape, grad_output.device)
        for param_idx in ctx.params_indices:
            fp16_param = ModelParameters.fp16_params[param_idx]
            free_storage(fp16_param.data)
        return grad_output, None


class AftForwardOffload(torch.autograd.Function):
    """
    A customized offload operation which forward is parameter release operation,
    backward is a parameter upload operation.

    Args:
        input_: input matrix.
        params_indices:.
    """

    @staticmethod
    def forward(ctx, input_, params_indices):
        # release
        ctx.params_indices = params_indices
        for param_idx in params_indices:
            free_storage(ModelParameters.fp16_params[param_idx].data)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # upload
        for param_idx in ctx.params_indices:
            fp16_param = ModelParameters.fp16_params[param_idx]
            alloc_storage(fp16_param.data)
            fp16_param.data.copy_(ModelParameters.fp32_master_params[param_idx].data)
        return grad_output, None



class PreBackwardPrefetch(torch.autograd.Function):
    """
    A customized prefetch operation which forward is parameter upload operation,
    backward is a parameter release operation.

    Args:
        input_: input matrix.
        params_indices:.
    """

    @staticmethod
    def forward(ctx, input_, params_indices, node_id):
        ctx.params_indices = params_indices
        ctx.node_id = node_id
        # nothing to run
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # prefetch
        params_indices = ctx.params_indices
        with torch.cuda.stream(GlobalCudaInfo.h2d_stream):
            for param_idx in params_indices:
                fp16_param = ModelParameters.fp16_params[param_idx]
                alloc_storage(fp16_param.data)
                fp16_param.data.copy_(ModelParameters.fp32_master_params[param_idx].data, non_blocking=True)

        # insert event to record H2D stream
        prefetch_event = torch.cuda.Event()
        prefetch_event.record(GlobalCudaInfo.h2d_stream)
        GlobalCudaInfo.param_event_map[ctx.node_id] = prefetch_event

        return grad_output, None, None

class AftForwardOffloadAsyn(torch.autograd.Function):
    """
    A customized offload operation which forward is parameter release operation,
    backward is a parameter upload operation.

    Args:
        input_: input matrix.
        params_indices:.
    """

    @staticmethod
    def forward(ctx, input_, params_indices, syn_upload_flag, node_id):
        # offload
        ctx.params_indices = params_indices
        ctx.syn_upload_flag = syn_upload_flag
        ctx.node_id = node_id
        for param_idx in params_indices:
            print("AftForwardOffloadAsyn", param_idx, ModelParameters.fp16_params[param_idx].data.shape)
            free_storage(ModelParameters.fp16_params[param_idx].data)
        return input_

    @staticmethod
    def backward(ctx, grad_output):

        # wait parameter prefetch
        prefetch_event = GlobalCudaInfo.param_event_map.get(ctx.node_id, None)
        if prefetch_event is not None:
            assert isinstance(prefetch_event, torch.cuda.Event)
            prefetch_event.wait()
            # torch.cuda.current_stream().wait_event(prefetch_event)
        elif ctx.syn_upload_flag:
            for param_idx in ctx.params_indices:
                fp16_param = ModelParameters.fp16_params[param_idx]
                alloc_storage(fp16_param.data)
                fp16_param.data.copy_(ModelParameters.fp32_master_params[param_idx].data)
        return grad_output, None, None, None

def convert_upload_to_action(tensor, params_indices):
    '''
    Convert UploadSpec into runtime action, implement upload operation target tensor.

    Argument:
        tensor(torch.Tensor): Tensor stored in each device, which could be different in different ranks.
    '''
    return PreForwardUpload.apply(tensor, params_indices)

def convert_offload_to_action(tensor, params_indices):
    '''
    Convert OffloadSpec into runtime action, implement offload operation target tensor.

    Argument:
        tensor(torch.Tensor): Tensor stored in each device, which could be different in different ranks.
    '''
    return AftForwardOffload.apply(tensor, params_indices)


def convert_prefetch_to_action(tensor, params_indices, node_id):
    '''
    Convert UploadSpec into runtime action, implement upload operation target tensor.

    Argument:
        tensor(torch.Tensor): Tensor stored in each device, which could be different in different ranks.
    '''
    return PreBackwardPrefetch.apply(tensor, params_indices, node_id)

def convert_offload_to_action_asyn(tensor, params_indices, syn_upload_flag=False, node_id=0):
    '''
    Convert OffloadSpec into runtime action, implement offload operation target tensor.

    Argument:
        tensor(torch.Tensor): Tensor stored in each device, which could be different in different ranks.
    '''
    return AftForwardOffloadAsyn.apply(tensor, params_indices, syn_upload_flag, node_id)


def replace_node_users(orig_node: Node, inserted_node: Node):
    user_list = list(orig_node.users.keys())
    for user in user_list:
        if user == inserted_node:
            continue
        new_args = list(user.args)
        new_kwargs = dict(user.kwargs)
        # the origin node may be a positional argument or key word argument of user node
        if orig_node in new_args:
            # substitute the origin node with offload_apply_node
            new_args[new_args.index(orig_node)] = inserted_node
            user.args = tuple(new_args)
        elif str(orig_node) in new_kwargs:
            # substitute the origin node with offload_apply_node
            new_kwargs[str(orig_node)] = inserted_node
            user.kwargs = new_kwargs


def runtime_offload_apply_pass(gm: torch.fx.GraphModule):
    """
    This pass is used to add the offload spec apply node to the origin graph.
    """
    mod_graph = gm.graph
    nodes = tuple(mod_graph.nodes)
    for node in nodes:
        if node.node_info.has_param:
            param_indices = node.node_info.param_indices
            assert isinstance(param_indices, list)

            last_inp_node = list(node._input_nodes.keys())[-1]
            # mod_graph.inserting_before(node) maybe invalid
            with mod_graph.inserting_after(last_inp_node):
                upload_apply_node = mod_graph.create_node('call_function', convert_upload_to_action,
                                                          args=(last_inp_node, param_indices))
            replace_node_users(last_inp_node, upload_apply_node)

            if node.node_info.offload_param_flag:
                with mod_graph.inserting_after(node):
                    offload_apply_node = mod_graph.create_node('call_function', convert_offload_to_action,
                                                               args=(node, param_indices))
                replace_node_users(node, offload_apply_node)
    return gm


def runtime_asyn_offload_apply_pass(gm: torch.fx.GraphModule):
    """
    This pass is used to add the asynchronous offload spec apply node to the origin graph.
    """
    mod_graph = gm.graph
    nodes = tuple(mod_graph.nodes)
    for node in nodes:

        node_to_prefetch = node.node_info.node_to_prefetch
        if node_to_prefetch is not None:
            param_indices = node_to_prefetch.node_info.param_indices
            node_id = node_to_prefetch.node_info.node_id
            assert isinstance(param_indices, list)
            with mod_graph.inserting_after(node):
                prefetch_apply_node = mod_graph.create_node('call_function', convert_prefetch_to_action,
                                                          args=(node, param_indices, node_id))
            replace_node_users(node, prefetch_apply_node)

        if node.node_info.has_param:
            param_indices = node.node_info.param_indices
            assert isinstance(param_indices, list)

            # last_inp_node = list(node._input_nodes.keys())[-1]

            def _extract_last_input_node(cur_node):
                for n in list(cur_node._input_nodes.keys()).__reversed__():
                    if n.op == "get_attr":
                        continue
                    return n

            last_inp_node = _extract_last_input_node(node)

            with mod_graph.inserting_after(last_inp_node):
                upload_apply_node = mod_graph.create_node('call_function', convert_upload_to_action,
                                                          args=(last_inp_node, param_indices))
            replace_node_users(last_inp_node, upload_apply_node)

            if node.node_info.offload_param_flag:
                syn_upload_flag = node.node_info.syn_upload_flag
                node_id = node.node_info.node_id
                with mod_graph.inserting_after(node):
                    offload_apply_node = mod_graph.create_node('call_function', convert_offload_to_action_asyn,
                                                               args=(node, param_indices, syn_upload_flag, node_id))
                replace_node_users(node, offload_apply_node)
    gm.graph.print_tabular()
    # print(len(ModelParameters.fp16_params), ModelParameters.param_idx)
    return gm

