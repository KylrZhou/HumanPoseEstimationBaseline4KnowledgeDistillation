import numpy as np
from torch.nn import Module

def HeatmapGenerate(anno, output_size, sigma = 2, keypoint_weight = None):
    if keypoint_weight is not None:
        keypoints = anno
    else:
        keypoints = anno['keypoints']
        keypoint_weight = anno['kweight']
    
    heatmap = []
    for i in range(len(keypoints)):
        if keypoint_weight[i] != 0: 
            coords = keypoints[i]
            x, y = np.meshgrid(np.arange(output_size[1]), np.arange(output_size[0]))
            dists = np.sqrt((x[:, :, np.newaxis] - coords[0])**2 +
                            (y[:, :, np.newaxis] - coords[1])**2)
            gaussians = np.exp(-dists**2 / (2 * sigma**2))
            tmp = np.sum(gaussians, axis=2)
            tmp /= np.max(tmp)
            heatmap.append(tmp)
        else:
            heatmap.append(np.zeros(output_size))
    heatmap = np.array(heatmap)
    return heatmap

class HeatmapGenerateC():
    def __init__(self, output_size, sigma = 2):
        self.output_size = output_size
        self.sigma = sigma
        
    def MAIN(batch_keypoints, batch_keypoint_weight = None):
        if batch_keypoint_weight is None:
            batch_keypoint_weight = np.ones(batch_keypoints[0:2])
        for num in range(len(batch_keypoints)):
            keypoints = batch_keypoints[num]
            keypoint_weight = keypoint_weight[num]
            if keypoint_weight is None:
                keypoint_weight = np.ones(len(keypoints))
            heatmap = []
            for i in range(len(keypoints)):
                if keypoint_weight[i] != 0: 
                    coords = keypoints[i]
                    x, y = np.meshgrid(np.arange(self.output_size[1]), np.arange(self.output_size[0]))
                    dists = np.sqrt((x[:, :, np.newaxis] - coords[0])**2 +
                                    (y[:, :, np.newaxis] - coords[1])**2)
                    gaussians = np.exp(-dists**2 / (2 * self.sigma**2))
                    tmp = np.sum(gaussians, axis=2)
                    tmp /= np.max(tmp)
                    heatmap.append(tmp)
                else:
                    heatmap.append(np.zeros(self.output_size))
            heatmap = np.array(heatmap)
            batch.append(heatmap)
        batch = np.array(batch)
        return batch