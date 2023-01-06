from typing import List
from functools import partial
import torch
import torch.nn as nn
from colossalai.tensor.param_op_hook import ColoParamOpHook
from colossalai.tensor.param_op_hook import ColoParamOpHookManager


class ParamUploadHook(ColoParamOpHook):

    def __init__(self) -> None:
        super().__init__()

    def pre_op(self, params):
        # move to cuda
        for p in params:
            p.data = p.data.to("cuda")

    def post_op(self, params):
        pass

    def pre_forward(self, params: List[torch.Tensor]) -> None:
        self.pre_op(params)

    def post_forward(self, params: List[torch.Tensor]) -> None:
        pass

    def pre_backward(self, params: List[torch.Tensor]) -> None:
        pass

    def post_backward(self, params: List[torch.Tensor]) -> None:
        pass


class GradOffloadHook():

    def __init__(self):
        self.grad_hook_list = []

    def grad_handle(self, p, grad):
        p.grad = p.grad.to('cpu')

    def register_grad_hook(self, module: torch.nn.Module):
        for p in module.parameters():
            if p.requires_grad:
                self.grad_hook_list.append(p.register_hook(partial(self.grad_handle, p)))

    def remove_grad_hook(self):
        for hook in self.grad_hook_list:
            hook.remove()

class BasicOffloadModule(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.param_upload_hook = ParamUploadHook()
        self.grad_offload_hook = GradOffloadHook()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _pre_forward(self):
        self.grad_hook.register_grad_hook(self.module)

    def forward(self, *args, **kwargs):
        self.model.zero_grad(set_to_none=True)
        self._pre_forward()
        with ColoParamOpHookManager.use_hooks(self.param_upload_hook):
            outputs = self.model(*args, **kwargs)
        return outputs

    def backward(self, loss):
        loss.backward()
        self._post_backward()

    def _post_backward(self):
        self.grad_hook.remove_grad_hook()
