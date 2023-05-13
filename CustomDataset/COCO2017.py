import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms.functional import pil_to_tensor

sys.path.append('../')
from utils import HeatmapGenerate

def serialanno2annoweight(ipt):
    kypt = [[ipt[0], ipt[1]],
           [ipt[3], ipt[4]],
           [ipt[6], ipt[7]],
           [ipt[9], ipt[10]],
           [ipt[12], ipt[13]],
           [ipt[15], ipt[16]],
           [ipt[18], ipt[19]],
           [ipt[21], ipt[22]],
           [ipt[24], ipt[25]],
           [ipt[27], ipt[28]],
           [ipt[30], ipt[31]],
           [ipt[33], ipt[34]],
           [ipt[36], ipt[37]],
           [ipt[39], ipt[40]],
           [ipt[42], ipt[43]],
           [ipt[45], ipt[46]],
           [ipt[48], ipt[49]]]
    wght = [ipt[2], ipt[5], ipt[8], ipt[11], ipt[14], ipt[17], ipt[20], ipt[23], ipt[26], ipt[29], ipt[32], ipt[35], ipt[38], ipt[41], ipt[44], ipt[47], ipt[50]]
    return kypt, wght

def ImgnKyptResizer(IMG=None, KYPT=None, WGHT=None, BBOX=None):
    left = BBOX[0] - 1
    up = BBOX[1] - 1
    right = BBOX[0] + BBOX[2] + 1
    bottom = BBOX[1] + BBOX[3] + 1
    if len(IMG.size) < 3:
        IMG = IMG.convert('RGB')
    IMG = IMG.crop((left, up, right, bottom))
    for i in range(len(WGHT)):
        KYPT[i][0] = float(format(KYPT[i][0] - left, '.1f'))
        KYPT[i][1] = float(format(KYPT[i][1] - up, '.1f'))
        if KYPT[i][0] > 0 and KYPT[i][0] < BBOX[2]:
            if KYPT[i][1] > 0 and KYPT[i][1] < BBOX[3]:
                pass
            else:
                KYPT[i][0] = 0
                KYPT[i][1] = 0
                WGHT[i] = 0
        else:
            KYPT[i][0] = 0
            KYPT[i][1] = 0
            WGHT[i] = 0
    return IMG, KYPT, WGHT

def COCO2017Collatefn(batch):
    img, anno = zip(*batch)
    kypt, wght = zip(*anno)
    kypt = torch.tensor(kypt, dtype = torch.float32)
    wght = torch.tensor(wght, dtype = torch.uint8)
    return torch.stack(img), (kypt, wght)

class COCO2017Keypoint(Dataset):
    """
    Params:
        PATH (str): The path to your COCO2017 train dataset annotation
        ANNMODE (bool): 0 stands for different format of annotation, 0 is coordinates and 1 is heatmap
        TRANSFORM (Object.albumentations): Data augment api, currently albumentations is only support library 
    """
    def __init__(self, PATH, TRANSFORM = None, HEATMAPGEN = None):
        super(COCO2017Keypoint, self).__init__()
        with open(PATH, 'r') as f:
            self.PATH = json.load(f)
        self.coco_api = COCO(annotation_file = os.path.join(self.PATH['ROOT'], self.PATH['TrainAnn']))
        self.img_path = os.path.join(self.PATH['ROOT'],self.PATH['TrainImg']) 
        self.anno_ids = list(self.coco_api.getAnnIds(catIds=1))
        self.transform = TRANSFORM
        self.HeatmapGen = HEATMAPGEN
    
    def __getitem__(self, index):
        anno_info = self.coco_api.loadAnns(self.anno_ids[index])[0]
        kypt, wght = serialanno2annoweight(anno_info['keypoints'])
        bbox = anno_info['bbox']
        img_info = self.img_path + '/' + self.coco_api.loadImgs(anno_info['image_id'])[0]['file_name']
        img, kypt, wght = ImgnKyptResizer(IMG = Image.open(img_info), KYPT = kypt, WGHT = wght, BBOX = bbox)
        if self.transform is not None:
            img = self.transform(image = np.array(img, dtype = np.float32), keypoints = kypt)
            kypt = img['keypoints']
            img = img['image'] 
            img = torch.tensor(img, dtype = torch.float32)
            if len(img.shape) == 1:
                img = torch.stack((img, img, img))
            img = img.permute(2, 0, 1)
        else:
            img = pil_to_tensor(img)
            if len(img.shape) == 1:
                img = torch.stack((img, img, img))
        if self.HeatmapGen is not None:
            kypt = self.HeatmapGen.MAIN(kypt, wght)
        anno = [kypt, wght]
        return img, anno
    
    def __len__(self):
        return len(self.anno_ids)