from functools import partial
import torch
import torch.nn as nn

from colossalai.fx import ColoTracer

class GradOffloadHook():

    def __init__(self):
        self.grad_hook_list = []
        self.grad_accs = []

    def grad_handle(self, grad):
        grad.data = grad.data.to("cpu")
        return grad

    def register_grad_hook(self, module: torch.nn.Module):
        for p in module.parameters():
            if p.requires_grad:
                self.grad_hook_list.append(p.register_hook(self.grad_handle))

    def remove_grad_hook(self):
        for hook in self.grad_hook_list:
            hook.remove()



class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2, bias=False)
        self.fc2 = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out

model = MyModel().cuda()
data = torch.rand((1, 2), device="cuda")
meta_args = {'x': data.to('meta')}
graph = ColoTracer().trace(model, meta_args=meta_args)
for node in graph.nodes:
    print(type(node._input_nodes.keys()))
    print(node.op)
    if len(list(node._input_nodes.keys())) > 0:
        print(type(list(node._input_nodes.keys())[0]))


grad_hook = GradOffloadHook()
grad_hook.register_grad_hook(model)
loss = torch.sum(model(data))
loss.backward()
print("fc1 weight", model.fc1.weight.data)
print("fc1 grad", model.fc1.weight.grad.device)
print("fc2 weight", model.fc2.weight.data)
print("fc2 grad", model.fc2.weight.grad.device)



