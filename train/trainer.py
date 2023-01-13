import os, sys
import torch
import torch.optim as optim

from utils.tools import *


class Trainer:
    def __init__(self, model, train_loader, eval_loader, hparam):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.max_batch = hparam['max_batch']
        self.epoch = 0
        self.iter = 0
        self.optimizer = optim.SGD(model.parameters(), lr=hparam['lr'], momentum=hparam['momentum'])
        
        self.scheduler_multistep = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=[20,40,60],
                                                                  gamma=0.5)

    def run(self):
        while True:
            self.model.train()
            # loss calculation
            
            self.epoch += 1
            