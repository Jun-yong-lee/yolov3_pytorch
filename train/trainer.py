import os, sys
import torch
import torch.optim as optim

from utils.tools import *
from train.loss import *


class Trainer:
    def __init__(self, model, train_loader, eval_loader, hparam, device):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.max_batch = hparam['max_batch']
        self.device = device
        self.epoch = 0
        self.iter = 0
        self.yololoss = Yololoss(self.device, self.model.n_classes)
        self.optimizer = optim.SGD(model.parameters(), lr=hparam['lr'], momentum=hparam['momentum'])
        
        self.scheduler_multistep = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=[20,40,60],
                                                                  gamma=0.5)

    def run_iter(self):
        for i, batch in enumerate(self.train_loader):
            # drop the batch when invalid values
            if batch is None:
                continue
            input_img, targets, anno_path = batch
            
            input_img = input_img.to(self.device, non_blocking=True)
            
            output = self.model(input_img)
            
            loss, loss_list = self.yololoss.compute_loss(output, targets, self.model.yolo_layers)
            
            # get loss between output and target
            
            print(f"output - len : {len(output)}, shape : {output[0].shape}")
        
    def run(self):
        while True:
            self.model.train()
            # loss calculation
            
            self.run_iter()
            self.epoch += 1
            

        