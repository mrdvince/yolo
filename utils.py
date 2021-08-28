import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch


def boxes_iou(box1, box2):
    # width and height of each bounding box
    width_box1 = box1[2]
    width_box2 = box2[2]
    height_box1 = box1[3]
    height_box2 = box2[3]

    # calculate the area of each bounding box
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2

    # calculate the width of the union of the two bounding boxes
    mx = min(box1[0] - width_box1 / 2.0, box2[0] - width_box2 / 2.0)
    Mx = max(box1[0] - width_box1 / 2.0, box2[0] - width_box2 / 2.0)
    union_width = Mx - mx

    # calculate the height of the union of the two bounding boxes
    my = min(box1[1] - height_box1 / 2.0, box2[1] - height_box2 / 2.0)
    My = max(box1[1] - height_box1 / 2.0, box2[1] - height_box2 / 2.0)
    union_height = My - my

    # calculate width and height of the area of intersection of the two bounding boxes
    intersection_width = width_box1 + width_box2 - union_width
    intersection_height = height_box1 + height_box2 - union_height

    # if boxes don't overlap IOU is zero
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0

    # calculate the area of intersection of the two bounding boxes
    intersection_area = intersection_height * intersection_width

    # area of the union of the two bounding boxes
    union_area = area_box1 + area_box2 - intersection_area

    # IOU
    iou = intersection_area / union_area
    return iou


# non-maximal suppression
def nms(boxes, iou_thresh):
    # if no boxes do nothing
    if len(boxes) == 0:
        return boxes
    # torch tensor to keep track of the detection confidence
    det_confs = torch.zeros(len(boxes))
    # detection confidence of each predicted bounding box
    for i in range(len(boxes)):
        det_confs[i] = boxes[i][4]
    # sort the indices of the bounding boxes by detection confidence
    _, sort_ids = torch.sort(det_confs, descending=True)

    # empty list to hold best bounding boxes after non-maximal suppression
    best_boxes = []

    # non-maximal suppression
    for i in range(len(boxes)):
        box_i = boxes[sort_ids[i]]
        # detection confidence not zero
        if box_i[4] > 0:
            best_boxes.append(box_i)
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sort_ids[j]]
                if boxes_iou(box_i, box_j) > iou_thresh:
                    box_j[4] = 0
    return best_boxes
