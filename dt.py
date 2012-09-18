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
    def __init__(self, left, right, dataPoints, attr=None, value=None):
        self.left = left
        self.right = right
        self.attr = attr
        self.value = value
        self.dataPoints = dataPoints
        
        locSums = {}

        for point in dataPoints:
            if point.location not in locSums:
                locSums[point.location] = 0
            locSums[point.location] += 1

        self.locSums = locSums
        self.entropy = getEntropy(locSums.values())
    def __repr__(self):
        return "Attribute: "+str(self.attr)+", Value: "+str(self.value)+" Entropy: "+str(self.entropy)

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
        
        for pair in attributeLists[attr]:
            value, loc = pair
            pValue, pLoc = prev
            if loc != pLoc:
                splitList.append((value + pValue) / float(2))
            prev = pair
        splits[attr] = sorted(list(set(splitList)))

    return splits

def splitNode(node, attr, value):
    lPoints = []
    rPoints = []
    rPoints = filter(lambda x: (x.stations[attr] >= value), node.dataPoints)
    lPoints = filter(lambda x: (x.stations[attr] < value), node.dataPoints)
    
    lChild = Node(None, None, lPoints)
    rChild = Node(None, None, rPoints)
    
    return Node(lChild, rChild, node.dataPoints)
	
def getChildrenEntropy(rootNode):
    # Return weighted sum of children entropies
    r = rootNode.right.entropy * (float(len(rootNode.right.dataPoints)) / len(rootNode.dataPoints))
    l = rootNode.left.entropy * (float(len(rootNode.left.dataPoints)) / len(rootNode.dataPoints))
    
    return r + l

def getBestSplit(node, splits, previousInfo):
    best_infoGain = -1 * sys.maxint
    best_splitNode = None
    for split_attr in splits:
        for split_value in splits[split_attr]:
            testSplit = splitNode(node, split_attr, split_value)
            testSplit.attr = split_attr
            testSplit.value = split_value
            info = getChildrenEntropy(testSplit)
            #sys.stdout.write(str(split_value)+",")
            if (previousInfo - info) > best_infoGain:
                best_infoGain = (previousInfo - info)
                best_splitNode = testSplit
                #print str(info - previousInfo) + "     " + str(testSplit)
    return best_splitNode

dataPoints = loadTrainingData()

node = Node(None, None, dataPoints)
print node.locSums
print node.entropy

splits = getSplits(getAttributeLists(dataPoints))

split = getBestSplit(node, splits, node.entropy)
print split