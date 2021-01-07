import cv2
import numpy as np
import os

def get_iou(seg_path_gt, seg_path_other):
    gt_image = np.array(cv2.imread(seg_path_gt, 0))
    gt_image = np.where(gt_image < 128, gt_image, 1).astype(np.int8)

    other_image = cv2.resize(cv2.imread(seg_path_other, 0), gt_image.shape)
    other_image = np.where(other_image < 128, other_image, 1).astype(np.int8)

    intersection = np.sum(other_image & gt_image)
    union = np.sum(other_image | gt_image)
    iou = intersection / union

    return iou

def compute_iou(seg_path_gt, seg_path_cam, seg_path_seg, seg_path_seg_hor, seg_path_seg_ver, seg_path_seg_rot):

    ###################################################
    # CAM
    iou_CAM = get_iou(seg_path_gt, seg_path_cam)

    # SEG
    iou_SEG = get_iou(seg_path_gt, seg_path_seg)

    # SEG Horizontal Flip
    iou_SEG_hor = get_iou(seg_path_gt, seg_path_seg_hor)

    # SEG Vertical Flip
    iou_SEG_ver = get_iou(seg_path_gt, seg_path_seg_ver)

    # SEG Rotation
    iou_SEG_rot = get_iou(seg_path_gt, seg_path_seg_rot)

    ###################################################

    return iou_CAM, iou_SEG, iou_SEG_hor, iou_SEG_ver, iou_SEG_rot


if __name__ == '__main__':
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for cls in classes:
        seg_path_gt = './data/test_seg/{}.png'.format(cls)       # ground-truth seg map
        seg_path_cam = './visualize/CAM/{}_seg.png'.format(cls)  # output seg map from CAM
        seg_path_seg = './visualize/SEG/{}_seg.png'.format(cls)  # output seg map from SEG
        seg_path_seg_hor = './visualize/SEG_HOR_FLIP/{}_seg.png'.format(cls)  # output seg horizontal flip map from SEG
        seg_path_seg_ver = './visualize/SEG_VER_FLIP/{}_seg.png'.format(cls)  # output seg vertical flip map from SEG
        seg_path_seg_rot = './visualize/SEG_ROTATION/{}_seg.png'.format(cls)  # output seg rotation map from SEG

        # iou_CAM, iou_SEG = compute_iou(seg_path_gt, seg_path_cam, seg_path_seg)
        iou_CAM, iou_SEG, iou_SEG_hor, iou_SEG_ver, iou_SEG_rot = compute_iou(seg_path_gt, seg_path_cam,
                                                                              seg_path_seg, seg_path_seg_hor,
                                                                              seg_path_seg_ver, seg_path_seg_rot)

        # print('Class: {} | CAM IoU: {:.3f} | SEG IoU: {:.3f}'.format(cls, iou_CAM, iou_SEG))
        print('Class: {} |\tCAM IoU: {:.3f}\t|\tSEG IoU: {:.3f}\t|\tSEG_hor IoU: {:.3f}\t|'
              '\t''SEG_ver IoU: {:.3f}\t|\tSEG_rot IoU: {:.3f}'
              .format(cls, iou_CAM, iou_SEG, iou_SEG_hor, iou_SEG_ver, iou_SEG_rot))
