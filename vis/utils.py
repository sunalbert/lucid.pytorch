import cv2
import numpy as np


# TODO: please add gray image support

def save_grad_img(grads, out_path):
    """
    Save gradients as img
    :param grads: gradients obtained from visulaziation model
    :param out_path: the path to save gradients image
    :return:
    """
    grads = grads - grads.min()
    grads /= grads.max()
    grads = np.transpose(grads, [1, 2, 0])
    grads = grads[:, :, ::-1]
    grads = np.uint8(grads * 255)[..., ::-1]
    grads = np.squeeze(grads)
    cv2.imwrite(out_path, grads)


def save_cam_img(img, grads, out_path):
    """
    save the activation map on the original img
    :param img: original image with three chanels (RGB) in range(0,1)
    :param grads: grads w.r.t input image
    :param out_path: the path to save the image
    :return:
    """
    grads = np.transpose(grads, [1, 2, 0])
    grads = grads[:, :, ::-1]

    heat_map = cv2.applyColorMap(np.uint8(grads * 255), cv2.COLORMAP_JET)
    heat_map = heat_map / 255

    cam = heat_map + img
    cam = cam / np.max(cam)

    cv2.imwrite(out_path, np.uint8(255 * cam))
