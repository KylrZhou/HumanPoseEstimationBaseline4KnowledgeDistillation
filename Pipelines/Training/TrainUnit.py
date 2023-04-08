from time import time
from torch import tensor
from tqdm import tqdm
def TrainUnit(Model,
              Dataset,
              Optimizer,
              Device,
              Criterion,
              Metric,
              GTPreProcess = None,
              PostProcess = None):
    # Model Mode Train
    Model.train()
    Timer1 = time()
    LOG = {'DATATIME' : 0, 'BATCHTIME' : 0}
    for idx, (img, anno) in enumerate(tqdm(Dataset)): 
        Timer2 = time()
        # Clac Data Time
        Timer1 = Timer2 - Timer1
        LOG['DATATIME'] += Timer1
        # Empty the Optimizer
        Optimizer.zero_grad()
        # Transfer Target Data to Proper Device
        img = img.to(Device)
        target = anno['keypoints']
        target = target.to(Device)
        target_weight = anno['kweights']
        target_weight = target_weight.to(Device)
        # Process GT
        if GTPreProcess is not None:
            #target = target.view(-1,17,2)
            target, target_weight = GTPreProcess.MAIN(target, target_weight)
        # Inference
        output = Model(img)
        # Process Output
        if PostProcess is not None:
            #output = output.view(-1,17,2)
            output, target_weight = PostProcess.MAIN(output, target_weight)
        # Clac & Record loss
        loss, loss_name = Criterion(output, target, target_weight)
        LOG['LOSS'] = {}
        for i in range(len(loss)):
            try:
                LOG['LOSS'][loss_name[i]] += loss[i]
            except:
                LOG['LOSS'][loss_name[i]] = loss[i]
        loss_back = 0
        for i in loss:
            loss_back += i
        """
        for k, v in loss.items():
            try:
                LOG['LOSS'][k] += v
            except:
                LOG['LOSS'][k] = v
        """
        # Loss Backpropagation
        loss_back.backward()
        # Clac & Record Acc
        acc, acc_name = Metric(output, target, target_weight)
        LOG['ACC'] = {}
        for i in range(len(acc)):
            try:
                LOG['ACC'][acc_name[i]] += acc[i]
            except:
                LOG['ACC'][acc_name[i]] = acc[i]
        """
        for k, v in acc.items():
            try:
                LOG['ACC'][k] += v
            except:
                LOG['ACC'][k] = v
        """
    # Update the Network with Optimizer
        Optimizer.step()
        # Clac Batch Time
        Timer1 = time()
        Timer1 = Timer2 - Timer1
        LOG['BATCHTIME'] += Timer1
        Timer1 = time()
    # Clac Epoch Average Data
    LOG['DATATIME'] /= len(Dataset)
    LOG['BATCHTIME'] /= len(Dataset)
    for k, v in LOG['LOSS'].items():
        LOG['LOSS'][k] /= len(Dataset)
    for k, v in LOG['ACC'].items():
        LOG['ACC'][k] /= len(Dataset)
    return LOG