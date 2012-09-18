import math
from operator import itemgetter
import sys


class DataPoint:
    def __init__(self, loc, stats):
        self.location = loc
        self.stations = stats

    def __repr__(self):
        return '\n' + str(self.stations) + '   ' + self.location

class Node:
    def __init__(self, left, right, dataPoints, attribute, value):
        self.left = left
        self.right = right
        self.dataPoints = dataPoints
        self.attr = attribute
        self.value = value
        
        locSums = {}

        for point in dataPoints:
            if point.location not in locSums:
                locSums[point.location] = 0
            locSums[point.location] += 1

        self.locSums = locSums
        self.entropy = getEntropy(locSums.values())

#Load training data

def loadTrainingData():
    f = open('wifi.train', 'ru')
         
    lines = f.readlines()
    data = []

    for line in lines:
        tokens = line.split(' ')
        allTokens = []
        for t in tokens:
            subdivision = t.split(':')
            for s in subdivision:
                if s != "":
                    allTokens.append(s)
        i = 0
        loc = ""
        stat = {}
        for token in allTokens:
            if i == (len(allTokens)-1):
                loc = token.replace('\n','')
            elif i % 2 == 1: # if odd
                stat[int(allTokens[i-1])] = int(allTokens[i])
            i += 1

        for i in range(0, 10):
            if i not in stat:
                stat[i] = -1 * sys.maxint
        data.append(DataPoint(loc, stat))

    return data

def getEntropy(numbers):
    sum = float(0)
    for n in numbers:
        sum += n

    totalEntropy = 0
    for n in numbers:
        if n == 0:
            continue
        nn = float(n)
        entropy = (nn/sum)*(-1 * math.log(nn/sum))
        totalEntropy += entropy

    return totalEntropy

def getAttributeLists(dataList):
    splitsForAttributes = {}

    for point in dataList:
        for station in point.stations:
            if station not in splitsForAttributes:
                splitsForAttributes[station] = []
            splitsForAttributes[station].append( (point.stations[station], point.location) )

    for list in splitsForAttributes:
        splitsForAttributes[list] = sorted(splitsForAttributes[list], key=itemgetter(0))

    return splitsForAttributes

def getSplits(attributeLists):
    splits = {}
    for attr in attributeLists:
        splitList = []
        prev = attributeLists[attr][0]
        
        print attributeLists[attr]
        for pair in attributeLists[attr]:
            value, loc = pair
            pValue, pLoc = prev
            if loc != pLoc:
                print value
                splitList.append((value + pValue) / float(2))
            prev = pair
        splits[attr] = list(set(splitList))

    return splits

def splitNode(node):
    lPoints = []
    rPoints = []
    rPoints = filter(lambda x: (x.stations[node.attr] >= node.value), node.dataPoints)
    lPoints = filter(lambda x: (x.stations[node.attr] < node.value), node.dataPoints)
    
    lChild = Node(None, None, lPoints, None, None)
    rChild = Node(None, None, rPoints, None, None)

    node.left = lChild
    node.right = rChild

    return node

def getChildrenEntropy(rootNode):
    # Return weighted sum of children entropies
    r = rootNode.right.entropy * (float(len(rootNode.right.dataPoints)) / len(rootNode.dataPoints))
    l = rootNode.left.entropy * (float(len(rootNode.left.dataPoints)) / len(rootNode.dataPoints))
    
    return r + l

   


dataPoints = loadTrainingData()
node = Node(None, None, dataPoints, 0, -80)
print node.locSums
print node.entropy

print getChildrenEntropy(splitNode(node))
