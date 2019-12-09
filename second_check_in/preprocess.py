import numpy as np
from skimage.io import imread
from skimage.color import rgb2lab, rgb2hsv
from sklearn.feature_extraction.image import extract_patches_2d
import os
import time
from random import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import model_config as config
from skimage.transform import resize
from tqdm import tqdm

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


def read_image(path, size, full_image):
    dir_list = os.listdir(path)


    image_size = size
    p_size = int(image_size / 2)
    assert p_size * 2 == image_size
    patch_per_image = 2

    train_image_patch_list = list()
    train_gt_pathc_list = list()

    test_image_patch_list = list()
    test_gt_pathc_list = list()

    shuffle(dir_list)

    it = tqdm(enumerate(dir_list), desc='Pre processing', total=len(dir_list))
    for ind, dir in it:
        raw_image = imread(os.path.join(path, dir, f'{dir}_Dermoscopic_Image', f'{dir}.bmp'))
        gt = imread(os.path.join(path, dir, f'{dir}_lesion', f'{dir}_lesion.bmp'))

        raw_image = resize(raw_image, [image_size, image_size, 3])
        gt = resize(gt, [image_size, image_size, 1])

        image = transform_to_SAB(raw_image)
        gt = gt.astype(np.bool).astype(np.uint8)
        if not full_image:
            if ind >= 20:
                randint = int(time.time())
                image_patches = extract_patches_2d(image, (p_size, p_size), max_patches=patch_per_image, random_state=randint)
                gt_patches = extract_patches_2d(gt, (p_size, p_size), max_patches=patch_per_image, random_state=randint)
                gt_patches = np.expand_dims(gt_patches, -1)


                train_image_patch_list.append(image_patches)
                train_gt_pathc_list.append(gt_patches)
            else:
                img2 = image.reshape([1] + list(image.shape))
                gt2 = gt.reshape([1] + list(gt.shape))
                for i in range(2):
                    for j in range(2):

                        test_image_patch_list.append(img2[:, i * p_size: (i + 1) * p_size, j * p_size: (j + 1) * p_size, :])
                        test_gt_pathc_list.append(gt2[:, i * p_size: (i + 1) * p_size, j * p_size: (j + 1) * p_size, :])
        else:
            if ind >= 20:
                train_image_patch_list.append(image.reshape([1] + list(image.shape)))
                train_gt_pathc_list.append(gt.reshape([1] + list(gt.shape)))
            else:
                test_image_patch_list.append(image.reshape([1] + list(image.shape)))
                test_gt_pathc_list.append(gt.reshape([1] + list(gt.shape)))


    train_image_patch_list = np.concatenate(train_image_patch_list, 0)
    train_gt_pathc_list = np.concatenate(train_gt_pathc_list, 0)

    test_image_patch_list = np.concatenate(test_image_patch_list, 0)
    test_gt_pathc_list = np.concatenate(test_gt_pathc_list, 0)

    num_test = test_gt_pathc_list.shape[0]

    train_data_gen = ImageDataGenerator()


    test_data_gen = ImageDataGenerator()

    config.eval_batch_size = num_test
    test_iterator = test_data_gen.flow(test_image_patch_list, test_gt_pathc_list, batch_size=num_test, shuffle=False)
    train_iterator = train_data_gen.flow(train_image_patch_list, train_gt_pathc_list, batch_size=config.batch_size, shuffle=True)

    num_train_steps = train_gt_pathc_list.shape[0] / config.batch_size
    num_test_steps = num_test

    return train_iterator, num_train_steps, test_iterator, num_test_steps
