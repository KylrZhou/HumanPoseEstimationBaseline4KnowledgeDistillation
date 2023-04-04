from time import time
def TrainUnit(Model,
              Dataset,
              Optimizer,
              Device,
              GTPreProcess = None,
              PostProcess = None,
              Criterion,
              Metric):
    # Model Mode Train
    Model.train()
    Timer1 = time()
    LOG = {}
    for idx, (img, anno) in enumerate(Dataset): 
        Timer2 = time()
        # Clac Data Time
        Timer1 = Timer2 - Timer1
        LOG['DATATIME'] += Timer1
        # Empty the Optimizer
        Optimizer.zero_grad()
        # Transfer Target Data to Proper Device
        target = anno['keypoints'].to(Device)
        target_weight = anno['kweights'].to(Device)
        # Process GT
        if GTPreProcess is not None:
            target, target_weight = GTPreProcess.MAIN(target, target_weight)
        # Inference
        output = Model(img)
        # Process Output
        if PostProcess is not None:
            output = PostProcess.MAIN(output)
        # Clac & Record loss
        loss = Criterion(output, target, target_weight)
        LOG['LOSS'] = {}
        for k, v in loss.items():
            try:
                LOG['LOSS'][k] += v
            except:
                LOG['LOSS'][k] = v
        # Loss Backpropagation
        loss.backward()
        # Clac & Record Acc
        acc = Metric(output, target, target_weight)
        LOG['ACC'] = {}
        for k, v in acc.items():
            try:
                LOG['ACC'][k] += v
            except:
                LOG['ACC'][k] = v
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
        LOG[k] /= len(Dataset)
    for k, v in LOG['ACC'].items():
        LOG[k] /= len(Dataset)
    return LOG