import json
import os
import mypy

import cv2
import numpy as np
import xmltodict

import torch
import copy

def parse_bboxes(annotation):
    bboxes = []
    class_mapping_table = {"neutral": 0,\
        "anger": 1,
        "surprise": 2,
        "smile": 3,
        "sad": 4}
    img_w = int(annotation['annotation']['size']['width'])
    img_h = int(annotation['annotation']['size']['height'])
    
    for object in annotation['annotation']['object']:
        pass

def xywh2xyxy(box_xywh: torch.Tensor) -> torch.Tensor: 
    box_xyxy = box_xywh.clone()
    box_xyxy[..., 0] = box_xywh[..., 0] - box_xywh[..., 2] / 2.
    box_xyxy[..., 1] = box_xywh[..., 1] - box_xywh[..., 3] / 2.
    box_xyxy[..., 2] = box_xywh[..., 0] + box_xywh[..., 2] / 2.
    box_xyxy[..., 3] = box_xywh[..., 1] + box_xywh[..., 3] / 2.
    return box_xyxy

def xyxy2xywh(box_xyxy: torch.Tensor) -> torch.Tensor: 
    box_xywh = box_xyxy.clone()
    box_xywh[..., 0] = (box_xyxy[..., 0] + box_xyxy[..., 2]) / 2.
    box_xywh[..., 1] = (box_xyxy[..., 1] + box_xyxy[..., 3]) / 2.
    box_xywh[..., 2] = box_xyxy[..., 2] - box_xyxy[..., 0]
    box_xywh[..., 3] = box_xyxy[..., 3] - box_xyxy[..., 1]
    return box_xywh

def iou_xyxy(boxA_xyxy: torch.Tensor, boxB_xyxy: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    """return area of intersection rectangle"""
    x11, y11, x12, y12 = torch.split(boxA_xyxy, 1, dim=1)
    x21, y21, x22, y22 = torch.split(boxB_xyxy, 1, dim=1)
    
    xA = torch.max(x11, x21.T)
    yA = torch.max(y11, x21.T)
    xB = torch.min(x12, x22.T)
    yB = torch.min(y12, y22.T)

    inter_area = (xB - xA).clamp(0) * (yB - yA).clamp(0)
    box_area_A = (x12 - x11) * (y12 - y11)
    box_area_B = (x22 - x21) * (y22 - y21)
    union_area = box_area_A + box_area_B.T - inter_area
    iou = inter_area / (union_area + eps)

    return iou

def iou_xywh(boxA_xywh, boxB_xywh):
    boxA_xyxy = xywh2xyxy(boxA_xywh)
    boxB_xyxy = xywh2xyxy(boxB_xywh)
    return iou_xyxy(boxA_xyxy, boxB_xyxy)

def iou_wh(boxA_wh, boxB_wh):
    eps = 1e-6
    
    w1, h1 = torch.split(boxA_wh, 1, dim=1)
    w2, h2 = torch.split(boxB_wh, 1, dim=1)
    
    innerW = torch.min(w1, w2.T).clamp(0)
    innerH = torch.min(h1, h2.T).clamp(0)

    inter_area = innerW * innerH
    boxA_area = w1 * h1
    boxB_area = w2 * h2
    iou = inter_area / (boxA_area + boxB_area.T + eps)
    return iou

def build_target_tensor(model, batch_pred_bboxes, batch_target_bboxes, input_size):
    batch_pred_bboxes = batch_pred_bboxes.cpu()
    batch_target_bboxes = copy.deepcopy(batch_target_bboxes)
    h, w = input_size
    o = (4 + 1 + model.num_classes)

    batch_size = len(batch_target_bboxes)
    batch_target_tensor = []
    for _ in range(batch_size):
        single_target_tensor = []
        for idx, stride in enumerate(model.strides):
            for _ in range(len(model.anchors_mask[idx])):
                single_target_tensor.append(torch.zeros((h // stride, w // stride, o), dtype=torch.float32))
        batch_target_tensor.append(single_target_tensor)
    
    for idx_batch in range(batch_size):
        single_target_bboxes = []
        for single_target_bbox in batch_target_bboxes[idx_batch]:
            c = int(torch.round(single_target_bbox[0]))

            bbox_xy = single_target_bbox[1:3].clone().view(1, 2)
            bbox_wh = single_target_bbox[3:].clone().view(1, 2)
            
            bbox_wh[0, 0] *= h
            bbox_wh[0, 1] *= w

            iou = iou_wh(bbox_wh, model.anchors_wh)
            iou, idx_yolo_layer = torch.max(iou, dim=-1)

            grid_h, grid_w = batch_target_tensor[idx_batch][idx_yolo_layer].shape[:2]

            grid_tx = bbox_xy[0, 0] * grid_w
            grid_ty = bbox_xy[0, 1] * grid_h

            idx_grid_tx = int(torch.floor(grid_tx))
            idx_grid_ty = int(torch.floor(grid_ty))

            if batch_target_tensor[idx_batch][idx_yolo_layer][idx_grid_ty, idx_grid_tx, 4] == 1:
                continue
            
            # tbd

def main():
    xyxy = xywh2xyxy(torch.tensor([0, 0, 10, 5]))
    print(xyxy)

if __name__ == '__main__':
    main()
