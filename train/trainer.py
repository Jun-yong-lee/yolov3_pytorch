import os, sys
import torch
import torch.optim as optim

from utils.tools import *
from train.loss import *

from tensorboardX import SummaryWriter

class Trainer:
    def __init__(self, model, train_loader, eval_loader, hparam, device, torch_writer):
        
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
        self.torch_writer = torch_writer

    def run_iter(self):
        for i, batch in enumerate(self.train_loader):
            # drop the batch when invalid values
            if batch is None:
                continue
            input_img, targets, anno_path = batch
            
            input_img = input_img.to(self.device, non_blocking=True)
            
            output = self.model(input_img)
            
            # get loss between output and target
            loss, loss_list = self.yololoss.compute_loss(output, targets, self.model.yolo_layers)
            
            # get gradients
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler_multistep.step(self.iter)
            self.iter += 1

            loss_name = ['total_loss', 'obj_loss', 'cls_loss', 'box_loss']
            
            if i % 10 == 0:
                print(f"epoch {self.epoch} / iter {self.iter} lr {get_lr(self.optimizer)} loss {loss.item()}")
                self.torch_writer.add_scalar('lr', get_lr(self.optimizer), self.iter)
                self.torch_writer.add_scalar('total_loss', loss, self.iter)
                for ln, lv in zip(loss_name, loss_list):
                    self.torch_writer.add_scalar(ln, lv, self.iter)
        return loss
                    
        
    def run(self):
        while True:
            self.model.train()
            # loss calculation
            loss = self.run_iter()
            self.epoch += 1
            
            # save model (checkpoint)
            checkpoint_path = os.path.join("./output", "model_epoch" + str(self.epoch) + ".pth")
            torch.save({'epoch' : self.epoch,
                       'iteration' : self.iter,
                       'model_state_dict' : self.model.state_dict(),
                       'optimizer_state_dict' : self.optimizer.state_dict(),
                       'loss' : loss}, checkpoint_path)
            

        