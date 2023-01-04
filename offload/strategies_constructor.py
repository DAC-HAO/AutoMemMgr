
import torch
from torch.fx import Graph, Node
from offload_strategy import OffloadStrategiesVector
from strategy_generator import StrategyGenerator
from options import SolverOption

class OffloadStrategiesConstructor:
    """
    OffloadStrategiesConstructor is used to construct the offload plan for the model execution.

    Args:
        graph (Graph): a Graph object used for analysis and strategy generation.
        solver_option (SolverOption): a SolverOptions object which specifies the preferences for plan searching.
    """

    def __init__(self, graph: Graph, solver_option: SolverOption):
        self.graph = graph
        assert graph.owning_module is not None, 'The given graph is not associated with a owning_module'
        self.root_module = self.graph.owning_module
        self.nodes = list(graph.nodes)
        self.leaf_strategies = []
        self.strategy_map = {}
        self.solver_option = solver_option
        self.no_strategy_nodes = []

    def build_strategies_and_cost(self):
        """
        This method is to build the strategy vector for each node in the computation graph.
        """

        def _check_no_strategy_for_node(node):
            label = False
            if node.op in ('placeholder', 'get_attr', 'call_method', 'output'):
                label = True

            elif node.op == "call_module":
                target = node.target
                submod = self.root_module.get_submodule(target)
                if (
                        len(list(submod.named_parameters(recurse=False))) == 0
                        and len(list(submod.named_buffers(recurse=False))) == 0
                ):
                    label = True

            elif node.op == "call_function":
                label = True
                for inp_node in node._input_nodes:
                    if inp_node.op == "get_attr":
                        label = False
                        break

            return label

        for node in self.nodes:
            node.meta["offload_param"] = False
            strategies_vector = OffloadStrategiesVector(node)

            if _check_no_strategy_for_node(node):
                self.no_strategy_nodes.append(node)
                pass

            generator = StrategyGenerator(node, self.graph)
            strategies_vector.extend(generator.generate())

            # TODO
