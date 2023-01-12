from ast import parse
import torch
import argparse
import os, sys

from utils.tools import *

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

def train():
    print("train")

def eval():
    print("evaluation")

def demo():
    print("demo")

if __name__ == "__main__":
    print("main")
    
    args = parse_args()
    
    # cfg parser
    module_defs = parse_hyperparm_config(args.cfg)
    print(module_defs)
    if args.mode == "train": # python main.py --gpus 0 --mode train --cfg yolov3_kitti.cfg
        # training
        train()
    elif args.mode == "eval":
        # evaluation
        eval()
    elif args.mode == "demo":
        # demo
        demo()
    else:
        print("unknown mode")
        
    print("finish")