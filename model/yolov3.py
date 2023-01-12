import os, sys
import numpy as np
import torch
import torch.nn as nn

from utils.tools import *

def make_conv_layer(layer_idx : int, modules : nn.Module, layer_info : dict, in_channel : int):
    filters = int(layer_info['filters']) # output channel
    size = int(layer_info['size']) # kernel size
    stride = int(layer_info['stride']) # stride
    pad = int(layer_info['pad']) # (size - 1) // 2
    modules.add_module('layer_' + str(layer_idx) + '_conv',
                       nn.Conv2d(in_channel, filters, size, stride, pad))
    
    if layer_info['batch_normalize'] == '1':
        modules.add_module('layer_' + str(layer_idx) + '_bn',
                           nn.BatchNorm2d(filters))
    if layer_info['activation'] == 'leaky':
        modules.add_module('layer_' + str(layer_idx) + '_act',
                           nn.LeakyReLU())
    if layer_info['activation'] == 'relu':
        modules.add_module('layer_' + str(layer_idx) + '_act',
                           nn.ReLU())  
        
def make_shortcut_layer(layer_idx : int, modules : nn.Module):
    modules.add_module('layer_' + str(layer_idx) + "_shortcut", nn.Identity())
      
def make_route_layer(layer_idx : int, modules : nn.Module):
    modules.add_module('layer_' + str(layer_idx) + "_route", nn.Identity())
    
def make_upsample_layer(layer_idx : int, modules : nn.Module, layer_info : dict):
    stride = int(layer_info['stride'])
    modules.add_module('layer_' + str(layer_idx) + '_upsample',
                       nn.Upsample(scale_factor=stride, mode='nearest'))

class YoloLayer(nn.Module):
    def __init__(self, layer_info : dict, in_width : int, in_height : int, is_train : bool):
        super(YoloLayer, self).__init__()
        self.n_classes = int(layer_info['classes'])
        self.ignore_thresh = float(layer_info['ignore_thresh'])
        self.box_attr = self.n_classes + 5 # box[4] + objectness[1] + class_prob[n_classes]
        mask_idxes = [int(x) for x in layer_info["mask"].split(",")]
        anchor_all = [int(x) for x in layer_info["anchors"].split(",")]
        anchor_all = [(anchor_all[i],anchor_all[i+1]) for i in range(0,len(anchor_all),2)]
        self.anchor = torch.tensor([anchor_all[x] for x in mask_idxes])
        self.in_width = in_width
        self.in_height = in_height
        self.stride = None
        self.lw = None
        self.lh = None
        self.is_train = is_train

    def forward(self, x):
        # x is input, [N C H W]
        self.lw, self.lh = x.shape[3], x.shape[2]
        self.anchor = self.anchor.to(x.device)
        self.stride = torch.tensor([torch.div(self.in_width, self.lw, rounding_mode='floor'),
                                    torch.div(self.in_height, self.lh, rounding_mode='floor')]).to(x.device)

class Darknet53(nn.Module):
    def __init__(self, cfg, param, training):
        super().__init__()
        self.batch = int(param['batch'])
        self.in_channels = int(param['in_channels'])
        self.in_width = int(param['in_width'])
        self.in_height = int(param['in_height'])
        self.n_classes = int(param['classes'])
        self.module_cfg = parse_model_config(cfg)
        self.module_list = self.set_layer(self.module_cfg)
        self.training = training

    def set_layer(self, layer_info):
        module_list = nn.ModuleList()
        in_channels = [self.in_channels] # First channels of input
        for layer_idx, info in enumerate(layer_info):
            modules = nn.Sequential()
            if info['type'] == "convolutional":
                make_conv_layer(layer_idx, modules, info, in_channels[-1])
                in_channels.append(int(info['filters']))
            elif info['type'] == 'shortcut':
                make_shortcut_layer(layer_idx, modules)
                in_channels.append(in_channels[-1])
            elif info['type'] == 'route':
                make_route_layer(layer_idx, modules)
                layers = [int(y) for y in info["layers"].split(",")]
                if len(layers) == 1:
                    in_channels.append(in_channels[layers[0]])
                elif len(layers) == 2:
                    in_channels.append(in_channels[layers[0]] + in_channels[layers[1]+1])
            elif info['type'] == 'upsample':
                make_upsample_layer(layer_idx, modules, info)
                in_channels.append(in_channels[-1])
            elif info['type'] == 'yolo':
                yololayer = YoloLayer(info, self.in_width, self.in_height, self.training)
                modules.add_module('layer_' + str(layer_idx) + '_yolo', yololayer)
                in_channels.append(in_channels[-1])
            
            module_list.append(modules)
        return module_list
