import torch
import torchvision
import numpy as np


class Extractor(object):
    def __init__(self, base_model, target_layers):
        super(Extractor, self).__init__()
        self.base_model = base_model
        self.target_layers = target_layers
        self.grads = []

    def save_graident(self, grad):
        self.grads.append(grad)

    def __call__(self, x):
        outs = []
        for name, module in self.base_model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_graident)
                outs.append(x)
        return outs, x






