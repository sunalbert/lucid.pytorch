
import torchvision.models as models



def inference():
    model = models.resnet18()
    for name, module in model._modules.items():
        if name == 'layer4':
            for n in module.children():
                print(n)


if __name__=='__main__':
    inference()