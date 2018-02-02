import torch
from torch.autograd import Function


class GuidedBackProRelu(Function):
    def forward(self, input):
        self.save_for_backward(input)
        return input.clamp(min=0)

    def backward(self, grad_outputs):
        input, = self.saved_tensors
        grad_input = grad_outputs.clone()
        grad_input[grad_outputs < 0] = 0
        grad_input[input < 0] = 0
        return grad_input

    def named_parameters(self, memo, submodule_prefix):
        return []


class DeconvnetRelu(Function):
    def forward(self, input):
        return input.clamp(min=0)

    def backward(self, grad_outputs):
        grad_input = grad_outputs.clone()
        grad_input[grad_outputs < 0] = 0
        return grad_input