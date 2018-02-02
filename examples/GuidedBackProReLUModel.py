import os
import cv2
import torch
import numpy as np
import torchvision.models as models
from torch.autograd import Variable

from vis.utils import save_cam_img, save_grad_img
from vis.algs import GuidedBackProReLUModel


os.environ['CUDA_VISIBLE_DEVICES'] = '7'

img_path = './imgs/cat_dog.png'

def preprocess(img):
    """
    Apply preprocess to input
    img
    :param processed_img: input image
    :return: torch tensor wrapping a img
    """
    means = np.asarray([0.485, 0.456, 0.406])
    stds = np.asarray([0.229, 0.224, 0.225])

    processed_img = img.copy()[:,:,::-1]
    processed_img -= means
    processed_img /= stds

    processed_img = np.transpose(processed_img, (2, 0, 1))
    processed_img = np.ascontiguousarray(processed_img)
    img_tensor = torch.from_numpy(processed_img)
    img_tensor.unsqueeze_(0)
    return img_tensor


def visualization():
    vis_model = GuidedBackProReLUModel(models.vgg19(pretrained=True), use_cuda=True)
    img = cv2.imread(img_path)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input_img = preprocess(img)
    input_img = Variable(input_img, requires_grad=True)
    mask = vis_model(input_img)
    save_grad_img(mask, 'grads.jpg')
    save_cam_img(img, mask, 'cam.jpg')


if __name__ == '__main__':
    visualization()