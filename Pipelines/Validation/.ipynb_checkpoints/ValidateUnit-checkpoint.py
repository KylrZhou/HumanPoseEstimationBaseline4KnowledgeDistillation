from time import time
from torch import no_grad

def ValidateUnit(Model,
                 Dataset,
                 Device,
                 Metric,
                 GTPreProcess = None,
                 PostProcess = None,
                 Criterion = None):
    # Model Mode Train
    Model.eval()
    with no_grad():
        Timer1 = time()
        LOG = {}
        for idx, (img, anno) in enumerate(Dataset): 
            Timer2 = time()
            # Clac Data Time
            Timer1 = Timer2 - Timer1
            LOG['DATATIME'] += Timer1
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
            if Criterion is not None:
                loss = Criterion(output, target, target_weight)
                try:
                    for k, v in loss.items():
                        LOG[k] += v
                except KeyError:
                    for k, v in loss.items():
                        LOG[k] = v
            # Clac & Record Acc
            acc = Metric(output, target, target_weight)
            loss = Criterion(output, target, target_weight)
            try:
                for k, v in acc.items():
                    LOG[k] += v
            except KeyError:
                for k, v in acc.items():
                    LOG[k] = v
            # Clac Batch Time
            Timer1 = time()
            Timer1 = Timer2 - Timer1
            LOG['BATCHTIME'] += Timer1
            Timer1 = time()
    # Clac Epoch Average Data 
    for k, v in LOG.items():
        LOG[k] /= len(Dataset)
    return LOG