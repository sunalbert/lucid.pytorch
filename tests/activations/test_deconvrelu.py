import nose
from nose.tools import with_setup
from nose.tools import assert_equal

import torch
import numpy as np
from torch.autograd import Variable
from vis.activations import DeconvnetRelu


def set_up():
    pass


def teardown():
    pass


@with_setup(set_up, teardown)
def test_deconvrelu_foward():
    pass


@with_setup(set_up, teardown)
def test_deconvrelu_backward():
    pass



if __name__ == '__main__':
    nose.run()
