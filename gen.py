import os
import cv2
import time
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from memory_profiler import profile
from pycocotools.coco import COCO as coco


PATH = []
with open('PATH.json','r') as f:
    PATH = json.load(f)
api = coco(annotation_file = os.path.join(PATH['ROOT'], PATH['ValAnn']))
ids = list(api.anns.keys())
anno = {}


def DATA_PATH_LOADING(index):
    #anno_dict = api.getAnnIds(imgIds = ids[index])
    print(ids[0])
    anno_dict = api.loadAnns(ids = ids[index])
    if type(anno_dict) == type([]):
        for k in anno_dict[0].keys():
            anno[k] = []
    else:
        for k in anno_dict.keys():
            anno[k] = []
    for i in anno_dict:
        for k ,v in i.items():
            anno[k].append(v)
    img = Image.open(os.path.join(PATH['ROOT'], PATH['ValImg'], api.loadImgs(ids = ids[index])[0]['file_name']))
    img = np.array(img)
    return img, anno


if __name__ == '__main__':
    start_t = time.time()

    #for i in range(2,100):
    img, anno = DATA_PATH_LOADING(0)
    #print(i)

    end_t = time.time()
    print(f'\n TIME USAGE: {end_t-start_t} \n')

    plt.imshow(img)
    #plt.show()
