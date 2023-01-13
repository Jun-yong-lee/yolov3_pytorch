from ast import parse
import torch
import argparse
import os, sys
from torch.utils.data.dataloader import DataLoader

from utils.tools import *
from dataloader.yolodata import *
from dataloader.data_transforms import *
from model.yolov3 import *
from train.trainer import *

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOV3_PYTORCH arguments")
    parser.add_argument("--gpus", type=int, nargs='+', default=[], help="List of GPU device id")
    parser.add_argument("--mode", type=str, help="mode : train / eval / demo", 
                        default=None)
    parser.add_argument("--cfg", type=str, help="model config path",
                        default=None)
    parser.add_argument("--checkpoint", type=str, help="model checkpoint path",
                        default=None)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def train(cfg_param = None, using_gpus = None):
    print("train")
    # dataloader 6881 images /batch : 4
    my_transform = get_transformations(cfg_param=cfg_param, is_train=True)
    
    train_data = Yolodata(is_train=True,
                        transform=my_transform,
                        cfg_param=cfg_param)
    train_loader = DataLoader(train_data,
                              batch_size=cfg_param['batch'],
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True)

    model = Darknet53(args.cfg, cfg_param, training=True)
    model.train()
    model.initialize_weights()

    train = Trainer(model=model, train_loader=train_loader, eval_loader=None, hparam=cfg_param)
    
    # for name, param in model.named_parameters():
    #     print(f"name : {name}, shape : {param.shape}")

def eval(cfg_param = None, using_gpus = None):
    print("evaluation")

def demo(cfg_param = None, using_gpus = None):
    print("demo")

if __name__ == "__main__":
    print("main")
    
    args = parse_args()
    
    # cfg parser
    net_data = parse_hyperparm_config(args.cfg)

    cfg_param = get_hyperparam(net_data)
    print(cfg_param)
    
    using_gpus = [int(g) for g in args.gpus]
    
    if args.mode == "train": # python main.py --gpus 0 --mode train --cfg yolov3_kitti.cfg
        # training
        train(cfg_param = cfg_param)
    elif args.mode == "eval":
        # evaluation
        eval(cfg_param = cfg_param)
    elif args.mode == "demo":
        # demo
        demo(cfg_param = cfg_param)
    else:
        print("unknown mode")
        
    print("finish")
    