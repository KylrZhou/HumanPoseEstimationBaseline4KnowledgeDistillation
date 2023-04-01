from tqdm import tqdm
from time import time
from torch import load, device
from torch.cuda import is_available
import sys

sys.path.append('../..')
from utils.model.ModelMerge import ModelMerge

def BasicTraining(Dataset=None,#
                  ValDataset=None,#
                  Backbone=None, BackbonePTH=None,#
                  Neck=None, NeckPTH=None,#
                  Head=None, HeadPTH=None,#
                  PostProcess=None,
                  GTPreProcess=None,
                  Epochs=None,#
                  Criterion=None,#
                  Metric4Train=None,#
                  Metric4Val=None,
                  Optimizer=None,#
                  Scheduler=None,#
                  Device=None,
                  LogInterval=None,
                  SaveInterval=None,
                  ValInterval=None,
                  log=None):
    print("\nTrain Pipeline:")
    if Device == None:
        Device = device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'    | Training Device is Assigned to {Device}')
    else:
        print(f'    | Training Device is Set to {Device}')
    
    if Dataset == None:
        raise Warning("Training Pipeline: No #Dataset# Assigned!")
        return
    else:
        print("    | Dataset Assigned!")
        
    if Metric4Train == None:
        raise Warning("Training Pipeline: No #Metric4Train# Assigned!")
        return
    else:
        print("    | Metric4Train Assigned!")
        
    if Metric4Val == None:
        print("    | Warning: No #Metric4Val# Assigned! Validation Metric Will Use #Metric4Train#")
        Metric4Val = Metric4Train
    else:
        print("    | Metric4Val Assigned!")
        
    if Backbone == None:
        raise Warning("Training Pipeline: No #Backbone# Assigned!")
        return
    else:
        if BackbonePTH != None:
            print("    | Backbone: Loading Trained Parameters", end = '')
            BackbonePTH = load(BackbonePTH)
            Backbone.load_state_dict(BackbonePTH)
            print(" -> DONE")
        else:
            print("    | Backbone: Assigned!")
            
    if Neck == None:
        print("    | No #Neck# Assigned!")
    else:
        if NeckPTH != None:
            print("    | Neck: Loading Trained Parameters", end = '')
            NeckPTH = load(NeckPTH)
            Neck.load_state_dict(NeckPTH)
            print(" -> DONE")
        else:
            print("    | Neck: Assigned!")
            
    if Head == None:
        print("    | No #Head# Assigned!")
    else:
        if HeadPTH != None:
            print("    | Head: Loading Trained Parameter", end = '')
            HeadPTH = load(HeadPTH)
            Head.load_state_dict(HeadPTH)
            print(" -> DONE")
        else:
            print("    | Head: Assigned!")
    
    IterSize = len(Dataset)
    Model = ModelMerge(Backbone, Neck, Head)
    timer = 0
    DataTime = 0
    BatchTime = 0
    for epoch in range(1, Epochs+1):
        Model.train()
        DataTime = time()
        for idx, (img, anno) in enumerate(Dataset): 
            timer = time()
            DataTime = timer - DataTime
            idx += 1
            LOG = {}
            LOG['MODE'] = 'TRAIN'
            LOG['EPOCH'] = epoch
            LOG['ITER'] = f'{idx}/{IterSize}'
            LOG['DATATIME'] = DataTime
            Optimizer.zero_grad()
            target = anno['keypoints'].to(Device)
            target_weight = anno['kweights'].to(Device)
            if GTPreProcess is not None:
                anno = GTPreProcess.MAIN(target, target_weight)
            output = Model(img)
            if PostProcess is not None:
                output = PostProcess.MAIN(output)
            loss = Criterion(output, target, target_weight)
            loss.backward()
            LOG['LOSS'] = loss
            acc = Metric4Train(output, target, target_weight)
            LOG['ACCURACY'] = acc
            Optimizer.step()
            BatchTime = time()
            BatchTime = timer - BatchTime
            LOG['BATCHTIME'] = BatchTime
            if idx % LogInterval == 0 or idx == IterSize:
                print('MODE: {} EPOCH: {} ITER: {} DATATIME: {} BATCHTIME: {} LOSS: {} ACCURACY: {}'.format(LOG['MODE'], LOG['EPOCH'], LOG['ITER'], LOG['DATATIME'], LOG['BATCHTIME'], LOG['LOSS'], LOG['ACCURACY']))
            #record data
            DataTime = time()
            break
        Scheduler.step()