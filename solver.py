import torch
from torch.fx.graph import Graph
from torch.fx.node import Node
from colossalai.utils.cuda import get_current_device
from colossalai.fx.profiler import (calculate_fwd_out, calculate_fwd_tmp, calculate_fwd_in, is_compatible_with_meta, parameter_size)
from strategies_constructor import OffloadStrategiesConstructor

class Solver:

    def __init__(self,
                 graph: Graph,
                 strategies_constructor: OffloadStrategiesConstructor,
                 memory_budget: float = -1.0):
        self.graph = graph
        self.strategies_constructor = strategies_constructor
        self.leaf_strategies = self.strategies_constructor.leaf_strategies
        self.nodes = [strategies_vector.node for strategies_vector in self.leaf_strategies]
        self.memory_budget = memory_budget if memory_budget > 0 \
            else torch.cuda.get_device_properties(get_current_device()).total_memory

    def _compute_mem_saving(self):
        peak_mem = 0
        total_mem_saving = 0

        # TODO 参数大小不应该初始化为整个模型，应该逐层的upload到gpu
        # param_size = parameter_size(self.graph.owning_module)

        # runtime_mem = param_size

        runtime_mem = 0

        for node in self.graph.nodes:
            runtime_mem = runtime_mem + calculate_fwd_tmp(node) + calculate_fwd_out(node)
            # prefetch parameter
            runtime_mem += node.node_info.param_size
            print(runtime_mem)
            total_mem_saving += min(node.node_info.runtime_fwd_mem - runtime_mem, 0)
            node.node_info.runtime_fwd_mem = runtime_mem

            peak_mem = max(runtime_mem, peak_mem)
            if node.node_info.offload_param_flag:
                runtime_mem -= node.node_info.param_size

        grad_in_computed = {}
        for node in self.graph.nodes.__reversed__():
            runtime_mem -= calculate_fwd_out(node)
            runtime_mem = runtime_mem + node.meta['bwd_mem_tmp'] + node.meta['bwd_mem_out']
            if node.node_info.has_param:
                if node.node_info.offload_param_flag:
                    # upload
                    runtime_mem += node.node_info.param_size
                # add weighted node gradient
                runtime_mem += node.node_info.param_size
                print(runtime_mem)
                total_mem_saving += min(node.node_info.runtime_bwd_mem - runtime_mem, 0)
                node.node_info.runtime_bwd_mem = runtime_mem

                peak_mem = max(runtime_mem, peak_mem)

                # release parameter and offload gradient
                runtime_mem -= 2 * node.node_info.param_size
            peak_mem = max(runtime_mem, peak_mem)
            runtime_mem = runtime_mem - node.meta['bwd_mem_tmp'] - calculate_fwd_tmp(node)

            # TODO 需要考虑有多个user node 的情况，当前只释放了一个bwd_out
            # release grad_in of current node
            for grad_in in node.meta["fwd_out"]:
                if isinstance(grad_in, torch.Tensor):
                    runtime_mem -= grad_in.numel() * grad_in.element_size()

            for in_node in list(node._input_nodes.keys()):
                # # release fwd_in (fwd_out) of current node (input nodes)
                # if calculate_fwd_out(in_node) > 0 and (not fwd_out_released[in_node]):
                #     runtime_mem -= calculate_fwd_out(in_node)
                #     fwd_out_released[in_node] = True

                # map multiple gradients of output to one tensor
                if grad_in_computed.get(in_node, False):
                    runtime_mem -= calculate_fwd_out(in_node)
                    grad_in_computed[in_node] = True

        return peak_mem, total_mem_saving

    def _call_solver_greedy_v1(self):
        peak_mem, total_mem_saving = self._compute_mem_saving()
        assert total_mem_saving == 0
        while peak_mem > self.memory_budget:
            offload_node = None
            max_profit = 0
            reduced_peak_mem = peak_mem
            for node in self.nodes:
                if (not node.node_info.offload_param_flag) and node.node_info.has_param:
                    node.node_info.offload_param_flag = True
                    tmp_peak_mem, tmp_total_mem_saving = self._compute_mem_saving()
                    profit = (peak_mem - tmp_peak_mem) / node.strategies_vector[0].comm_cost
                    if profit > max_profit:
                        offload_node = node
                        max_profit = profit
                        reduced_peak_mem = tmp_peak_mem
                    node.node_info.offload_param_flag = False
            offload_node.node_info.offload_param_flag = True
            peak_mem = reduced_peak_mem

