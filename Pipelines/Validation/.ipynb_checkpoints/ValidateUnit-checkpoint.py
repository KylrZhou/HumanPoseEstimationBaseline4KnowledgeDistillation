import torch

def coordreverse(kypt, ratio, padding, padding_type, bbox):
    #kypt =
    pass
    
def ValidateUnit(Model=None,
                 Dataset=None):
    #Model.eval()
    cocoDt = []
    for idx, (img, anno) in enumerate(Dataset):
        with torch.no_grad():
            img = img.to('cuda')
            #img = Model(img)
            imid = anno['image_id']
            wght = anno['wght']
            ratio = anno['ratio']
            padding = anno['padding']
            padding_type = anno['padding_type']
            for i in range(img.shape[0]):
                img_info = {'image_id': 0, 'category_id': 1, 'keypoints': 0, 'score':0}
                img_info['image_id'] = imid[i]
                img_info['score'] = wght[i].sum()/34
                #img_info['keypoints'] = 
    return 