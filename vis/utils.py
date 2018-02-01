import cv2
import numpy as np


def save_grad_img(grads):
    grads = grads - grads.min()
    grads /= grads.max()

    grads = np.transpose(grads, [1, 2, 0])
    grads = grads[:, :, ::-1]
    grads = np.uint8(grads * 255)[..., ::-1]
    cv2.imwrite('grads_img.jpg', grads)


def save_cam_img(img, mask):
    heat_map = cv2.applyColorMap(np.uint8(mask * 255), cv2.COLORMAP_JET)
    cv2.imwrite('heat_map.jpg', heat_map)
    heat_map = heat_map / 255
    cam = heat_map + img
    cam = cam / np.max(cam)
    return cam
