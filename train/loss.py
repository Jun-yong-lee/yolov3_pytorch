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
        
        # pout.shape : [batch, anchors, grid_y, grid_x, box_attrib]
        # the number of boxes in each yolo layer : anchors * grid_h * grid_w
        # yolo0 -> 3 * 19 * 19, yolo1 -> 3 * 38 *38, yolo2 -> 3 * 76 * 76
        # total boxes : 22,743.
        # positive prediction vs negative prediction
        # pos : neg = 0.01 : 0.99
        # Only in positive prediction, we can get box_loss and class_loss
        # in negative prediction, only obj_loss
        
        # get positive targets
        tcls, tbox, tindices, tanchors = self.get_targets(pred, targets, yololayer)
        
        
        # 3 yolo layers
        for pidx, pout in enumerate(pred):
            batch_id, anchor_id, gy, gx = tindices[pidx]
            
            tobj = torch.zeros_like(pout[...,0], device=self.device)
            
            num_targets = batch_id.shape[0]
            
            if num_targets:
                ps = pout[batch_id, anchor_id, gy, gx] # [batch, anchor, grid_h, grid_w, box_attrib]
                pxy = torch.sigmoid(ps[...,0:2])
                pwh = torch.exp(ps[...,2:4]) * tanchors[pidx]
                pbox = torch.cat((pxy, pwh), 1)
                # assignment
                iou = bbox_iou(pbox, tbox[pidx])

            

            
    def get_targets(self, preds, targets, yololayer):
        num_anc = 3
        num_targets = targets.shape[0]
        tcls, tboxes, indices, anch = [], [], [], []
        
        gain = torch.ones(7, device=self.device)   
        
        # anchor_index
        ai = torch.arange(num_anc, device=targets.device).float().view(num_anc, 1).repeat(1, num_targets)
        # targets shape : [batch_id, class_id, box_cs, box_cy, box_w, box_h, anchor_id]
        targets = torch.cat((targets.repeat(num_anc, 1, 1), ai[:, :, None]), 2)
        
        for yi, yl in enumerate(yololayer):
            anchors = yl.anchor / yl.stride
            
            gain[2:6] = torch.tensor(preds[yi].shape)[[3,2,3,2]] # grid_w, grid_h / ex) 0.1 -> 16.2 real box location
            
            t = targets * gain
            
            if num_targets:
                
                r = t[:, :, 4:6] / anchors[:, None]
                
                # select thresh ratios less than 4
                j = torch.max(r, 1./r).max(2)[0] < 4
                
                t = t[j]
            else:
                t = targets[0]
            
            # batch_id, class_id
            b, c = t[:, :2].long().T
            
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            
            gij = gxy.long()
            gi, gj = gij.T
            
            # anchor index
            a = t[:, 6].long()
            
            # add indices
            indices.append((b, a, gj.clamp(0, gain[3] - 1).long(), gi.clamp(0, gain[2] - 1).long()))
            # indices.append((b, a, gj.clamp(0, gain[3] - 1), gi.clamp(0, gain[2] - 1)))
            
            # add target_box
            tboxes.append(torch.cat((gxy - gij, gwh),dim=1))
            
            # add anchor
            anch.append(anchors[a])
            
            # add class
            tcls.append(c)
            
        return tcls, tboxes, indices, anch