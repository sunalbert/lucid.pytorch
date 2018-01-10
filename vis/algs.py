import torch
from torch.autograd import Variable
from vis.activations import GuidedBackProRelu


def replace_relu(module):
    for idx, m in module._modules.items():
        if m.__class__.__name__ == 'ReLU':
            module._modules[idx] = GuidedBackProRelu()
        else:
            replace_relu(m)


class GuidedBackProModel(object):
    def __init__(self, model, use_cuda):
        self.model = model.eval()
        if use_cuda:
            self.model = self.model.cuda()
        self.cuda = use_cuda

    def __call__(self, x, index=None):
        if self.cuda:
            out = self.model(Variable(x, requires_grad=True).cuda(), )
        else:
            out = self.model(Variable(x))
        out = out.view(-1)

        if index is None:
            index = int(torch.max(out))
        one_hot_mask = torch.zeros(out.size())
        one_hot_mask[index] = 1
        one_hot_mask = Variable(one_hot_mask)
        if self.cuda:
            one_hot_mask = torch.sum(one_hot_mask.cuda() * out)
        else:
            one_hot_mask = torch.sum(one_hot_mask * out)

        # backpropagation
        self.model.zero_grad()
        one_hot_mask.backward(retain_graph=True)

        result = x.grad.data.cpu().numpy()
        return result[0]


class VanillaBackProModel(object):
    def __init__(self, model, use_cuda):
        self.model = model.eval()
        if use_cuda:
            self.model = self.model.cuda()
        self.cuda = use_cuda

    def __call__(self, x, index=None):
        if self.cuda:
            out = self.model(Variable(x, requires_grad=True).cuda(), )
        else:
            out = self.model(Variable(x))
        out = out.view(-1)

        if index is None:
            index = int(torch.max(out))
        one_hot_mask = torch.zeros(out.size())
        one_hot_mask[index] = 1
        one_hot_mask = Variable(one_hot_mask)
        if self.cuda:
            one_hot_mask = torch.sum(one_hot_mask.cuda() * out)
        else:
            one_hot_mask = torch.sum(one_hot_mask * out)

        # backpropagation
        self.model.zero_grad()
        one_hot_mask.backward(retain_graph=True)

        result = x.grad.data.cpu().numpy()
        return result[0]


