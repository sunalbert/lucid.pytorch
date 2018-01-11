import torch
import torchvision
import numpy as np


class Extractor(object):
    def __init__(self, base_model, target_layers):
        super(Extractor, self).__init__()
        self.base_model = base_model
        self.target_layers = target_layers
        self.grads = []

    def save_grads(self, grad):
        self.grads.append(grad)

    def get_grads(self):
        return self.grads

    def __call__(self, x):
        inter_outs = []
        self.grads = []
        for name, module in self.base_model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_grads)
                inter_outs.append(x)
        return inter_outs, x






