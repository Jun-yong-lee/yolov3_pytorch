import numpy as np
import cv2
import torch
from torchvision import transforms as tf

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from utils.tools import *

def get_transformations(cfg_param = None, is_train = None):
    if is_train:
        data_transform = tf.Compose([AbsoluteLabels(),
                                     RelativeLabels()])
    else:
        data_transform = tf.Compose([AbsoluteLabels(),
                                     RelativeLabels()]) 
    return data_transform

# absolute bbox
class AbsoluteLabels(object):
    def __init__(self,):
        pass

    def __call__(self, data):
        image, label = data
        h, w, _ = image.shape
        label[:, [1, 3]] *= w # cx, w
        label[:, [2, 4]] *= h # cy, h
        return image, label

class RelativeLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        image, label = data
        h, w, _ = image.shape
        label[:, [1, 3]] /= w # cx, w
        label[:, [2, 4]] /= h # cy, h
        return image, label
