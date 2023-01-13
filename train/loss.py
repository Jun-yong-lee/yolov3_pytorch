import torch
import torch.nn as nn
from utils.tools import *
import os, sys

class Yololoss(nn.Module):
    def __init__(self, device, num_class):
        super(Yololoss, self).__init__()
        self.device = device
        self.num_class = num_class
        
    def compute_loss(self, pred, targets, yololayer):
        lcls, lbox, lobj = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        
        # get positive targets
        self.get_targets(pred, targets, yololayer)
        
        
        # 3 yolo layers
        for pidx, pout in enumerate(pred):
            print(f"yolo {pidx}, shape {pout.shape}")
            
            # pout.shape : [batch, anchors, grid_y, grid_x, box_attrib]
            # the number of boxes in each yolo layer : anchors * grid_h * grid_w
            # yolo0 -> 3 * 19 * 19, yolo1 -> 3 * 38 *38, yolo2 -> 3 * 76 * 76
            # total boxes : 22,743.
            # positive prediction vs negative prediction
            # pos : neg = 0.01 : 0.99
            # Only in positive prediction, we can get box_loss and class_loss
            # in negative prediction, only obj_loss
            
    def get_targets(self, preds, targets, yolo_layer):
        num_anc = 3
        num_targets = targets.shape[0]
        tcls, tboxes, indices, anch = [], [], [], []
        
        gain = torch.ones(7, device=self.device)   
        
        # anchor_index
        ai = torch.arange(num_anc, device=targets.device).float().view(num_anc, 1).repeat(1, num_targets)
        # targets shape : [batch_id, class_id, box_cs, box_cy, box_w, box_h, anchor_id]
        targets = torch.cat((targets.repeat(num_anc, 1, 1), ai[:, :, None]), 2)
        
        for yi, yl in enumerate(yolo_layer):
            anchors = yl.anchor / yl.stride
            print(anchors)
            
            gain[2:6] = torch.tensor(preds[yi].shape)[[3,2,3,2]] # grid_w, grid_h / ex) 0.1 -> 16.2 real box location
            
            t = targets * gain
            
            print(t)