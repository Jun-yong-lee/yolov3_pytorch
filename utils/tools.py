import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import tqdm
import torch
import torchvision

# parse model configuration
def parse_model_config(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
        
    module_defs = []
    type_name = None
    for line in lines:
        if line.startswith('['):
            type_name = line[1:-1].rstrip()
            if type_name == "net":
                continue
            module_defs.append({})        
            module_defs[-1]['type'] = type_name
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            if type_name == "net":
                continue
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs
    

# parse the yolov3 configuration
def parse_hyperparm_config(path):
    file = open(path, 'r')
    lines = file.read().split("\n")
    lines = [x for x in lines if x and not x.startswith("#")]
    lines = [x.rstrip().lstrip() for x in lines]
    
    module_defs = []
    for line in lines:
        if line.startswith("["):
            type_name = line[1:-1].rstrip()
            if type_name != "net":
                continue
            module_defs.append({})
            module_defs[-1]['type'] = type_name
            if module_defs[-1]['type'] == "convolutional":
                module_defs[-1]["batch_normalize"] = 0
        else:
            if type_name != "net":
                continue
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
            
    return module_defs

def get_hyperparam(cfg):
    for d in cfg:
        if d['type'] == 'net':
            batch = int(d['batch'])
            subdivision = int(d['subdivisions'])
            momentum = float(d['momentum'])
            decay = float(d['decay'])
            saturation = float(d['saturation'])
            lr = float(d['learning_rate'])
            burn_in = int(d['burn_in'])
            max_batch = int(d['max_batches'])
            lr_policy = d['policy']
            in_width = int(d['width'])
            in_height = int(d['height'])
            in_channels = int(d['channels'])
            classes = int(d['class'])
            ignore_class = int(d['ignore_cls'])

            return {'batch':batch,
                    'subdivision':subdivision,
                    'momentum':momentum,
                    'decay':decay,
                    'saturation':saturation,
                    'lr':lr,
                    'burn_in':burn_in,
                    'max_batch':max_batch,
                    'lr_policy':lr_policy,
                    'in_width':in_width,
                    'in_height':in_height,
                    'in_channels':in_channels,
                    'classes':classes,
                    'ignore_class':ignore_class}
        else:
            continue
        
def xywh2xyxy_np(x : np.array):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2 # minx
    y[..., 1] = x[..., 1] - x[..., 3] / 2 # miny
    y[..., 2] = x[..., 0] + x[..., 2] / 2 # maxx
    y[..., 3] = x[..., 1] + x[..., 3] / 2 # maxy
    return y

def drawBox(img):
    img = img * 255
    
    if img.shape[0] == 3:
        img_data = np.array(np.transpose(img, (1, 2, 0)), dtype=np.uint8)
        img_data = Image.fromarray(img_data)
    
    # draw = ImageDraw.Draw(img_data)
    
    plt.imshow(img_data)
    plt.show()

# box_a, box_b IOU
def bbox_iou(box1, box2, xyxy=False, eps=1e-9):
    box2 = box2.T

    if xyxy:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
        b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
        b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    # intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # union Area
    b1_w, b1_h = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    b2_w, b2_h = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = b1_w * b1_h + b2_w * b2_h - inter + eps
    
    iou = inter / union
    
    return iou

def cxcy2minmax(box):
    y = box.new(box.shape)
    xmin = box[..., 0] - box[..., 2] / 2
    ymin = box[..., 1] - box[..., 3] / 2
    xmax = box[..., 0] + box[..., 2] / 2
    ymax = box[..., 1] + box[..., 3] / 2
    
    y[..., 0] = xmin
    y[..., 1] = ymin
    y[..., 2] = xmax
    y[..., 3] = ymax
    return y

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def non_max_suppression(prediction, conf_thresh=0.1, iou_thresh=0.1):
    # num of class
    nc = prediction.shape[2] - 5
    
    # setting
    max_wh = 4096
    max_det = 300
    max_nms = 30000
    
    output = [torch.zeros((0, 6), device='cpu')] * prediction.shape[0]
    
    for xi, x in enumerate(prediction):
        x = x[x[..., 4] > conf_thresh]
        
        if not x.shape[0]:
            continue
        
        x[:, 5:] *= x[:, 4:5] # class *= objectness
        
        box = cxcy2minmax(x[:, :4])
        
        conf, j = x[:, 5:].max(1, keepdim=True)
        print(conf, j)
        x = torch.cat((box, conf, j.float()), dim=1)[conf.view(-1) > conf_thresh]
        
        # number of boxes
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        
        c = x[:, 5:6] * max_wh
        
        boxes, scores = x[:, :4] + c, x[:, 4]
        
        i = torchvision.ops.nms(boxes, scores, iou_thresh)
        
        if i.shape[0] > max_det:
            i = i[:max_det]
            
        output[xi] = x[i].detach().cpu()
    return output