import math

def EuclideanDistance(x, y):
    return math.sqrt((x[0]-y[0])^2 + (x[1] - y[1])^2)

def AvgDist(output, target, target_weight):
    dist = 0
    for i in range(len(output)):
        dist += EuclideanDistance(output[i], target[i]) * target_weight[i]
    dist/=len(output)
    return dist