import math
from torch import stack

def EuclideanDistance(x, y):
    return math.sqrt((x[0]-y[0])^2 + (x[1] - y[1])^2)

"""
def AvgDist(batch_output, batch_target, batch_target_weight):
    batch_dist = 0
    for num in range(len(batch_output)):
        output = batch_output[num]
        target = batch_target[num]
        target_weight = batch_target_weight[num]
        dist = 0
        for i in range(len(output)):
            dist += EuclideanDistance(output[i], target[i]) * target_weight[i]
        dist/=len(output)
        batch_dist += dist
    batch_dist/=len(batch_output)
    return [batch_dist], ['AvgDist']
"""

def AvgDist(input, target, weight):
    weight = stack((weight, weight), dim = -1)
    x = input - target
    x = x ** 2
    x = weight * x
    x = x.sum()
    x = x/weight.sum()
    return [x], ['AvgDist']