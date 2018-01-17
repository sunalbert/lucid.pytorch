import nose
from nose.tools import assert_equal
from nose.tools import with_setup

import torch
import numpy as np
from torch.autograd import Variable
from vis.activations import GuidedBackProRelu


def set_up():
    print("Test start")


def tear_down():
    print("Test done")


@with_setup(set_up, tear_down)
def test_forward():
    x = Variable(torch.randn(2,3))

    grelu = GuidedBackProRelu()
    out = grelu(x)

    x_mask = torch.clamp(x, min=0)
    res = torch.sum(x_mask - out)
    res = res.data.cpu().numpy()
    assert_equal(res, 0)


@with_setup(set_up, tear_down)
def test_backward():

    x = Variable(torch.randn(2,3), requires_grad=True)

    grelu = GuidedBackProRelu()
    out = grelu(x)
    out = torch.sum(out)
    out.backward()
    grad = x.grad

    x_mask = torch.gt(x, 0).float()
    result = torch.sum(x_mask - grad.float())
    result = result.data.numpy()

    assert_equal(result, 0)


if __name__=='__main__':
    nose.run()

