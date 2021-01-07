import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import ImageGrid


def get_images(path):
    item_names = os.listdir(path)

    for index, item in enumerate(item_names):
        item_names[index] = item[:-4]

    img_array = []
    for index in range(len(item_names)):
        im = cv2.cvtColor(cv2.imread(f'{path}/{item_names[index]}.png'), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (256, 256))
        img_array.append(im)
    img_array = np.array(img_array)

    return img_array


def get_images_double(path):
    item_names = os.listdir(path)

    for index, item in enumerate(item_names):
        item_names[index] = item[:-4]

    real_array = []
    mask_array = []
    for index in range(len(item_names)):
        img_name = f'{path}/{item_names[index]}.png'
        im = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (256, 256))
        if item_names[index][-3:] == 'seg':
            mask_array.append(im)
        else:
            real_array.append(im)
    real_array = np.array(real_array)
    mask_array = np.array(mask_array)

    return real_array, mask_array


gt_im_dir = 'data/test_seg_source'
gt_seg_dir = 'data/test_seg'

cam_dir = 'visualize/CAM'
seg_dir = 'visualize/SEG'
seg_hor_dir = 'visualize/SEG_HOR_FLIP'
seg_ver_dir = 'visualize/SEG_VER_FLIP'
seg_rot_dir = 'visualize/SEG_ROTATION'

gt_real = get_images(gt_im_dir)
gt_mask = get_images(gt_seg_dir)

cam_real, cam_mask = get_images_double(cam_dir)
seg_real, seg_mask = get_images_double(seg_dir)
seg_hor_real, seghor_hor_mask = get_images_double(seg_hor_dir)
seg_ver_real, seghor_ver_mask = get_images_double(seg_ver_dir)
seg_rot_real, seghor_rot_mask = get_images_double(seg_rot_dir)

img_grid_1 = np.zeros((60, 256, 256, 3), dtype=np.uint8)

counter = 0
for index in range(10):
    img_grid_1[counter] = gt_real[index]
    counter += 1
    img_grid_1[counter] = gt_mask[index]
    counter += 1
    img_grid_1[counter] = cam_real[index]
    counter += 1
    img_grid_1[counter] = cam_mask[index]
    counter += 1
    img_grid_1[counter] = seg_real[index]
    counter += 1
    img_grid_1[counter] = seg_mask[index]
    counter += 1

num_elements = len(img_grid_1)
titles = ['Source', 'SEG GT', 'CAM CAM', 'CAM SEG', 'SEG CAM', 'SEG SEG']

plt.figure(figsize=(6, 10))  # specifying the overall grid size
for i in range(num_elements):
    plt.subplot(10, 6, i + 1)
    plt.axis('off')
    plt.imshow(img_grid_1[i])
    if i < 6:
        plt.title(titles[i])
plt.show()


img_grid_2 = np.zeros((80, 256, 256, 3), dtype=np.uint8)

counter = 0
for index in range(10):
    img_grid_2[counter] = gt_real[index]
    counter += 1
    img_grid_2[counter] = gt_mask[index]
    counter += 1
    img_grid_2[counter] = seg_hor_real[index]
    counter += 1
    img_grid_2[counter] = seghor_hor_mask[index]
    counter += 1
    img_grid_2[counter] = seg_ver_real[index]
    counter += 1
    img_grid_2[counter] = seghor_ver_mask[index]
    counter += 1
    img_grid_2[counter] = seg_rot_real[index]
    counter += 1
    img_grid_2[counter] = seghor_rot_mask[index]
    counter += 1

num_elements = len(img_grid_2)
titles = ['Source', 'SEG GT', 'HOR_CAM', 'HOR_SEG', 'VER_CAM', 'VER_SEG', 'ROT_CAM', 'ROT_SEG']

plt.figure(figsize=(8, 10))  # specifying the overall grid size
for i in range(num_elements):
    plt.subplot(10, 8, i + 1)
    plt.axis('off')
    plt.imshow(img_grid_2[i])
    if i < 8:
        plt.title(titles[i])
plt.show()