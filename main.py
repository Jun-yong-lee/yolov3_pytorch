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

from tensorboardX import SummaryWriter

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

def collate_fn(batch):
    batch = [data for data in batch if data is not None]
    # skip invalid data
    if len(batch) == 0:
        return
    
    imgs, targets, anno_path = list(zip(*batch))
    
    imgs = torch.stack([img for img in imgs])
    
    if targets[0] is None or anno_path[0] is None:
        return imgs, None, None
    
    for i, boxes in enumerate(targets):
        # insert index of batch
        boxes[:,0] = i
        # print(boxes)
        
    targets = torch.cat(targets, 0)
    
    return imgs, targets, anno_path

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
                              shuffle=True,
                              collate_fn=collate_fn)
    
    eval_transform = get_transformations(cfg_param=cfg_param, is_train=False)
    eval_data = Yolodata(is_train=False,
                         transform=eval_transform,
                         cfg_param=cfg_param)
    eval_loader = DataLoader(eval_data,
                             batch_size=cfg_param['batch'],
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False,
                             collate_fn=collate_fn)

    model = Darknet53(args.cfg, cfg_param)
    model.train()
    model.initialize_weights()

    print(f"GPU : {torch.cuda.is_available()}")
    # set device
    if torch.cuda.is_available():
        # device = torch.device("cuda:0")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model = model.to(device)
    
    # load checkpoint
    # if checkpoint is existed, load the previous checkpoint
    # python main.py --mode train --cfg yolov3_kitti640480.cfg \
    # --checkpoint C:/study/yolo_data/yolov3_kitti640480_pretrained/model_epoch6.pth   
    if args.checkpoint is not None:
        print("load pretrained model ", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # for key, value in checkpoint['model_state_dict'].items():
        #     print(key, value)
        model.load_state_dict(checkpoint['model_state_dict'])
        # strict=False -> solve Unexpected key(s) error
        
    torch_writer = SummaryWriter("./output")
 
    trainer = Trainer(model=model, train_loader=train_loader, eval_loader=eval_loader, hparam=cfg_param, device=device, torch_writer=torch_writer)
    trainer.run()
    
    # tensorboard --logdir=./output --port 8888
    
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
    