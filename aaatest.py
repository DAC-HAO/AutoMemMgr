
import torch
import torch.nn as nn

from colossalai.fx import ColoTracer

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 1024, bias=False)
        self.fc2 = nn.Linear(1024, 1024, bias=False)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out

model = MyModel()
meta_args = {'x': torch.rand((1, 1024)).to('meta')}
graph = ColoTracer().trace(model, meta_args=meta_args)
for node in graph.nodes:
    print(type(node._input_nodes))
    print(node._input_nodes)
