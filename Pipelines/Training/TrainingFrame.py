from tqdm import tqdm
from time import time
from torch import load, device
from torch.cuda import is_available
import sys

sys.path.append('../..')
from utils.model.ModelMerge import ModelMerge

def TrainingFrame(Dataset=None,#
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
                  log_dir=None):
    # Variable Check
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

    Model = ModelMerge(Backbone, Neck, Head)
    
    # Train Start
    
    loggr = loggr(Write2Path = log_dir)
    
    for epoch in range(1, Epochs+1):
        
        # Train Function
        
        LOG = TrainUnit(Model = Model,
                        Dataset = Dataset,
                        Optimizer = Optimizer,
                        Device = Device,
                        GTPreProcess = GTPreProcess,
                        PostProcess = PostProcess,
                        Criterion = Criterion,
                        Metric = Metric4Train)
        
        Scheduler.step()
        
        loggr.update('MODE', 'Train')
        loggr.update('EPOCH', epoch)
        
        for k, v in LOG.items():
            loggr.update(k, v)
        
        loggr.wrt()
        
        if epoch % ValInterval == 0:
            LOG = TrainUnit(Model = Model,
                            Dataset = Dataset,
                            Device = Device,
                            GTPreProcess = GTPreProcess,
                            PostProcess = PostProcess,
                            Metric = Metric4Val)
            
            loggr.update('MODE', 'Val')
            loggr.update('EPOCH', epoch)
        
            for k, v in LOG.items():
                loggr.update(k, v)
        
            loggr.wrt()
            
        