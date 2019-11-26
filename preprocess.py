import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2lab, rgb2hsv
import os


def S_of_HSV(image):
    image_in_hsv = rgb2hsv(image)
    S_channel = image_in_hsv[:, :, 1]
    return S_channel


def A_of_LAB(image):
    image_in_lab = rgb2lab(image)
    A_channel = image_in_lab[:, :, 1]
    return A_channel


def B_of_RGB(image):
    B_channel = image[:, :, -1]
    return B_channel


def value_scale(array):

    scaled = (array - np.min(array)) / (np.max(array) - np.min(array))
    return scaled


def transform_to_SAB(image):

    S = S_of_HSV(image)
    A = A_of_LAB(image)
    B = B_of_RGB(image)

    S = value_scale(S)
    A = value_scale(A)
    B = value_scale(B)

    transformed = np.dstack((S, A, B))

    return transformed


def read_image(path):
    dir_list = os.listdir(path)

    image_list = list()
    gt_list = list()

    for dir in dir_list:
        raw_image = imread(os.path.join(path, dir, f'{dir}_Dermoscopic_Image', f'{dir}.bmp'))
        gt = imread(os.path.join(path, dir, f'{dir}_lesion', f'{dir}_lesion.bmp'))

        raw_image = resize(raw_image, [256, 256, 3])
        image = transform_to_SAB(raw_image)
        gt = resize(gt, [256, 256, 1]).astype(np.bool).astype(np.uint8)

        image_list.append(image)
        gt_list.append(gt)
    image_array = np.asarray(image_list)
    gt_array = np.asarray(gt_list)

    return image_array, gt_array










