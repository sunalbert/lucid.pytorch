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
        self.grads = []

        for 


