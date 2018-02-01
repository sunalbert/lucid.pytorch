import torch
import numpy as np
from torch.autograd import Variable
from vis.activations import GuidedBackProRelu
from vis.extractor import Extractor


def replace_relu(module):
    """
    Replace all the ReLU activation function
    with GuidedBackProRelu
    :param module:
    :return:
    """
    for idx, m in module._modules.items():
        if m.__class__.__name__ == 'ReLU':
            module._modules[idx] = GuidedBackProRelu()
        else:
            replace_relu(m)


class VanillaBackProModel(object):
    """
    Vanilla Backpropagation Model for
    visualiazation
    """
    def __init__(self, model, use_cuda):
        self.model = model.eval()
        if use_cuda:
            self.model = self.model.cuda()
        self.cuda = use_cuda

    def __call__(self, x, index=None):
        """
        Perform vanilla backpropagation visualization model
        :param x: input Variable
        :param index:
        :return:
        """
        if self.cuda:
            out = self.model(x.cuda())
        else:
            out = self.model(x)
        # out = out.view(-1)

        if index is None:
            if self.cuda:
                index = int(torch.max(out).data.cpu().numpy())
            else:
                index = int(torch.max(out).data.numpy()) # TODO: could be omitted

        # one_hot_mask = torch.zeros(out.size())
        # one_hot_mask[index] = 1
        one_hot_mask = np.zeros((1, out.size()[-1]), dtype=np.float32)
        one_hot_mask[0][index] = 1
        one_hot_mask = Variable(torch.from_numpy(one_hot_mask))

        # one_hot_mask = Variable(one_hot_mask, requires_grad=True)

        if self.cuda:
            one_hot_mask = torch.sum(one_hot_mask.cuda() * out)
        else:
            one_hot_mask = torch.sum(one_hot_mask * out)

        # backpropagation
        # self.model.zero_grad()
        one_hot_mask.backward()
        result = x.grad.data.cpu().numpy()
        return result[0]


class GuidedBackProModel(object):
    def __init__(self, model, use_cuda):
        self.model = model.eval()
        if use_cuda:
            self.model = self.model.cuda()
        self.cuda = use_cuda

    # TODO: unify the type of x: tensor or variable
    def __call__(self, x, index=None):
        if self.cuda:
            out = self.model(Variable(x, requires_grad=True).cuda(), )
        else:
            out = self.model(Variable(x))
        out = out.view(-1) # TODO: now the model only support classification task

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


# TODO: Now the GradCam only support visualize one layer at the same time
class GradCam(object):
    """GradCam visualization technique
    """
    def __init__(self, model, target_layers, use_cuda):
        assert len(target_layers) == 1

        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        else:
            self.model = model
        self.model.eval()

        self.extractor = Extractor(model, target_layers)

    def __call__(self, x, index=None):
        """ The implementation of GradCam

        :param x:
        :param index:
        :return:
        """
        assert x.size(0) == 1

        if self.cuda:
            inter_outs, final_out = self.extractor(x.cuda())
        else:
            inter_outs, final_out = self.extractor(x)

        final_out = final_out.view(-1)
        if index == None:
            index = int(torch.max(final_out))

        one_hot_mask = torch.zeros(final_out.size())
        one_hot_mask[index] = 1
        one_hot_mask = Variable(one_hot_mask)

        if self.cuda:
            one_hot_mask = torch.sum(one_hot_mask.cuda() * final_out)
        else:
            one_hot_mask = torch.sum(one_hot_mask * final_out)

        self.model.zero_grad()
        one_hot_mask.backward(retain_variables=True)

        target_grads = self.extractor.get_grads()







