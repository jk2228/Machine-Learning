import math
import sys


def func(x):
    return math.exp(-(x*x))

def getAverage(datapoints):
    sum = 0
    for x,y in datapoints:
        sum += y
    return float(sum)/len(datapoints)

def getImpurity(datapoints):
    average = getAverage(datapoints)
    sum = float(0)
    
    for x,y in datapoints:
        sum += math.pow((y - average), 2)
        
    return float(sum)/float(len(datapoints))
    
def getPoints(x1, x2):    
    diff = x2 - x1
    step = float(diff)/1000
    points = []
    
    for i in range(0, 1000):
        x = x1+(i*step)
        y = func(x)
        points.append((x,y))
    
    return points

def bestSplit(points):
    bestpurity = sys.maxint
    bestsplit = -1
    
    for i, _ in enumerate(points):
        split = i
        left = points[:split]
        right = points[split:]
        if len(left)==0 or len(right)==0: continue

        lImp = getImpurity(left)
        rImp = getImpurity(right)
        impurity = (float(len(left))/float(len(points)))*lImp + (float(len(right))/float(len(points)))*rImp
        
        if impurity < bestpurity:
            bestpurity = impurity
            bestsplit = split
    
    return bestsplit
    
allPoints = getPoints(-5, 5)
split = bestSplit(allPoints)

print allPoints[split]

left_1 = allPoints[:split]
right_1 = allPoints[split:]

split_l1 = bestSplit(left_1)
split_r1 = bestSplit(right_1)

print left_1[split_l1]
print right_1[split_r1]