from vis.algs import GuidedBackProRelu
import torchvision.models as models


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


def inference():
    model = models.vgg19()
    replace_relu(model)
    for name, module in model._modules.items():
        print(module)


if __name__=='__main__':
    inference()