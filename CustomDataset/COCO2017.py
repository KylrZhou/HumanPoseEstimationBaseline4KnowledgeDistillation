import os
import cv2
import json
import numpy as np
from PIL import Image
#import albumentations as A
from pycocotools.coco import COCO
import torchvision.transforms as T
from torch.utils.data import Dataset
#from albumentations.pytorch.transforms import ToTensorV2

class COCO2017Keypoint(Dataset):
    def __init__(self, PATH, transforms = None, mode = 'Train'):
        super(COCO2017Keypoint, self).__init__()
        with open(PATH,'r') as f:
            self.PATH = json.load(f)
        if mode == 'Train':
            self.api = COCO(annotation_file = os.path.join(self.PATH['ROOT'], self.PATH['TrainAnn']))
            self.img_path = os.path.join(self.PATH['ROOT'],self.PATH['TrainImg']) 
        elif mode == 'Val':
            self.api = COCO(annotation_file = os.path.join(self.PATH['ROOT'], self.PATH['ValAnn']))
            self.img_path = os.path.join(self.PATH['ROOT'],self.PATH['ValImg']) 
        else:
            raise Warning(f'Cannot Recognize Dataset Mode! \'{mode}\' Detected. It should be \'Train\' or \'Val\'')
        self.ids = list(self.api.anns.keys())
        self.transforms = transforms
    
    def __getitem__(self, index):
        anno_tmp = self.api.loadAnns(ids = self.ids[index])[0]
        anno = {'keypoints' : [], 'kweights' : []}
        bbox = []
        bbox.append(int(anno_tmp['bbox'][0]))
        bbox.append(int(anno_tmp['bbox'][1]))
        bbox.append(int(anno_tmp['bbox'][0] + anno_tmp['bbox'][2]))
        bbox.append(int(anno_tmp['bbox'][1] + anno_tmp['bbox'][3]))
        num_of_keypoints = int(len(anno_tmp['keypoints'])/3)
        for i in range(num_of_keypoints):
            tmp = [anno_tmp['keypoints'][i*3],anno_tmp['keypoints'][i*3+1], anno_tmp['keypoints'][i*3+2]]
            if tmp[2] == 0:                   
                anno['keypoints'].append([0, 0])
                anno['kweights'].append(0)
            elif tmp[0] > bbox[0] and tmp[0] < bbox[2] and tmp[1] > bbox[1] and tmp[1] < bbox[3]:
                tmp[0] -= bbox[0]
                tmp[1] -= bbox[1]
                anno['keypoints'].append([tmp[0], tmp[1]])
                anno['kweights'].append(tmp[2])
            else:                
                anno['keypoints'].append([0, 0])
                anno['kweights'].append(0)
        img = Image.open(os.path.join(self.img_path, 
                                      self.api.loadImgs(ids = anno_tmp['image_id'])[0]['file_name']))
        if img.mode == 'L':
            img = img.convert('RGB')
        img = np.array(img.crop((bbox[0], bbox[1], bbox[2], bbox[3])))
        if self.transforms is not None:
            #######
            try:
                transformed = self.transforms(image = img, keypoints = anno['keypoints'])
                img = transformed['image']
                anno['keypoints'] = transformed['keypoints']
            except:
                with open('/root/autodl-tmp/debug.txt', 'w') as f:
                    f.write(str(bbox[0]))
                    f.write(' , ')
                    f.write(str(bbox[1]))
                    f.write(' , ')
                    f.write(str(bbox[2]))
                    f.write(' , ')
                    f.write(str(bbox[3]))
                    f.write('\n')
                    f.write(os.path.join(self.img_path, self.api.loadImgs(ids = anno_tmp['image_id'])[0]['file_name']))
                return
        ToTensor = T.ToTensor()
        Normalize = T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        img = ToTensor(img)
        #img = Normalize(img)
        return img, anno
    
    def __len__(self):
        return len(self.ids)