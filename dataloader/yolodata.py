import torch
from torch.utils.data import Dataset

class Yolodata(Dataset):
    # formal path
    file_dir = ""
    anno_dir = ""
    file_txt = ""
    
    # train_dataset_path
    train_dir = "C:\\study\\yolo_data\\KITTI\\training"
    train_txt = "training.txt"
    valid_dir = "C:\\study\\yolo_data\\KITTI\\eval"
    valid_txt = "eval.txt"
    class_str = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
    num_class = None
    img_data = []
    
    def __init__(self, is_train=True, transform=None, cfg_param=None):
        super(Yolodata, self).__init__()
        self.is_train = is_train
        self.transform = transform
        self.num_class = cfg_param['class']
        
        if self.is_train:
            self.file_dir = self.train_dir + "\\JPEGImages\\"
            self.anno_dir = self.train_dir + "\\Annotations\\"
            self.file_txt = self.train_dir + "\\ImageSets\\" + self.train_txt
        else:
            self.file_dir = self.valid_dir + "\\JPEGImages\\"
            self.anno_dir = self.valid_dir + "\\Annotations\\"
            self.file_txt = self.valid_dir + "\\ImageSets\\" + self.valid_txt

        img_names = []
        img_data = []
        with open(self.file_txt, 'r', encoding='UTF-8', errors='ignore') as f:
            img_names = [ i.replace("\n", "") for i in f.readlines()]
        for i in img_names:
            if os.path.exists(self.file_dir + i + ".jpg"):
                img_data.append(i+".jpg")
            elif os.path.exists(self.file_dir + i + ".JPG"):
                img_data.append(i+".JPG")
            elif os.path.exists(self.file_dir + i + ".png"):
                img_data.append(i+".png")
            elif os.path.exists(self.file_dir + i + ".PNG"):
                img_data.append(i+".PNG")
        print("data len : {}".format(len(img_data)))
        self.img_data = img_data