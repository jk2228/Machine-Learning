import math
from operator import itemgetter
import sys
import random


class DataPoint:
    def __init__(self, loc, stats):
        self.location = loc
        self.stations = stats

    def __repr__(self):
        return '\n' + str(self.stations) + '   ' + self.location

class Node:
    def __init__(self, left, right, dataPoints, attr=None, value=None, label=None):
        self.left = left
        self.right = right
        self.attr = attr
        self.value = value
        self.dataPoints = dataPoints
        self.label = label # Only leaf nodes have labels
        
        locSums = {}

        for point in dataPoints:
            if point.location not in locSums:
                locSums[point.location] = 0
            locSums[point.location] += 1

        self.locSums = locSums
        self.entropy = getEntropy(locSums.values())
    def __repr__(self):
        return "Attribute: "+str(self.attr)+", Value: "+str(self.value)+" Entropy: "+str(self.entropy)
        
    def getDepth(self):
        if self.left == None and self.right == None:
            return 0
        v1 = self.left.getDepth()
        v2 = self.right.getDepth()
        return max(v1, v2)+1

#Load training data

def loadData(filename):
    f = open(filename, 'ru')
    
    #used to remove duplicates
    duplicateSet = {}
         
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
                
        if frozenset(stat.items()) not in duplicateSet:
            duplicateSet[frozenset(stat.items())] = 1
        else:
            continue
        
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

def findBoundSplits(value, splitList):
    startIndex = -1
    endIndex = -1
    started = False
    i = 0
    for pair in splitList:
        val, loc = pair
        if val == value:
            started = True
            if startIndex == -1:
                startIndex = i
        else:
            if started:
                endIndex = i
                started = False
        i += 1
        
    v1 , _ = splitList[startIndex]
    v2 , _ = splitList[endIndex]
    return (v1, v2)
    
def getSplits(attributeLists):
    splits = {}
    
    for attr in attributeLists:
        splitList = []
        prev = attributeLists[attr][0]
        
        for pair in attributeLists[attr]:
            value, loc = pair
            pValue, pLoc = prev
            if loc != pLoc:
                if value == pValue:
                    startSplit, endSplit = findBoundSplits(value, attributeLists[attr])
                    splitList.append(startSplit)
                    splitList.append(endSplit)
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

def getBestSplit(node, splits=None, out=False):
    if splits == None:
        splits = getSplits(getAttributeLists(node.dataPoints))
    if out:
        print ' ------------------- '
        print splits
    
    best_infoGain = -1 * sys.maxint
    best_splitNode = None
    previousInfo = node.entropy
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
                
    if best_splitNode.left.dataPoints == [] or best_splitNode.right.dataPoints == []:
        print "0 size"
        print best_splitNode.dataPoints
        if not out: getBestSplit(node, out=True)
    return best_splitNode
    
def getMajorityNode(dataPoints):
    allLabelFreqs = {}
    for p in dataPoints:
        if p.location not in allLabelFreqs:
            allLabelFreqs[p.location] = 0
        allLabelFreqs[p.location] += 1
    bestLabel = None
    highFreq = 0
    for l in allLabelFreqs:
        if allLabelFreqs[l] > highFreq:
            highFreq = allLabelFreqs[l]
            bestLabel = l
    tree = Node(None, None, dataPoints, label=bestLabel)
    return tree
    
''' Actually build the tree '''
def buildTree(dataPoints):
    homogenous = True
    loc = dataPoints[0].location
    for p in dataPoints:
        if p.location != loc:
            homogenous = False
            break
    if homogenous:
        return Node(None, None, dataPoints, label=loc)
    
    rootNode = Node(None, None, dataPoints)
    rootNode = getBestSplit(rootNode)
    
    if len(rootNode.left.dataPoints) == 0 or len(rootNode.right.dataPoints) == 0:
        return getMajorityNode(rootNode.dataPoints)
    
    rootNode.left = buildTree(rootNode.left.dataPoints)
    rootNode.right = buildTree(rootNode.right.dataPoints)

    return rootNode
    
def classifyPoint(tree, point):
    if tree.label != None:
        return tree.label
    
    if point.stations[tree.attr] >= tree.value:
        return classifyPoint(tree.right, point)
    else:
        return classifyPoint(tree.left, point)
        
def classifyDataSet(tree, dataset, out=True):
    total = 0
    incorrectTags = {}
    totalTags = {}
    
    i = 0
    for p in dataset:
        real = p.location
        tag = classifyPoint(tree, p)
        
        if real not in totalTags:
            totalTags[real] = 0
        totalTags[real] += 1
        
        
        
        if real != tag:
            total += 1
            if real not in incorrectTags:
                incorrectTags[real] = 0
            incorrectTags[real] += 1
        
        #print str(i)+".  Value: "+real+"   Guessed: "+tag
        
        i += 0
    
    if not out:
        return float(total)/len(dataset)
    
    print
    print '----------------------------------------'
    print
    print ' Location error:   '
    print
    for t in totalTags:
        if t not in incorrectTags:
            print t+':   \t'+str(0)
        else:
            print t+':   \t'+str(float(incorrectTags[t])/totalTags[t])
    print
    print '----------------------------------------'
    print
    print ' Number Misclassified: '+str(total)
    print ' Total data points:    '+str(len(dataset))
    print ' Total Error:  '+str(float(total)/len(dataset))
    print 
    
    return float(total)/len(dataset)
    
def compareAlgorithms(tree1, tree2, dataset):
    total = 0
    incorrectPointsA1 = []
    correctPointsA1 = []
    incorrectPointsA2 = []
    correctPointsA2 = []
    
    for p in dataset:
        real = p.location
        tag1 = classifyPoint(tree1, p)
        tag2 = classifyPoint(tree2, p)
        
        if tag1 != real:
            incorrectPointsA1.append(p)
        else:
            correctPointsA1.append(p)
            
        if tag2 != real:
            incorrectPointsA2.append(p)
        else:
            correctPointsA2.append(p)
        
    bothCorrect = 0
    A1notA2 = 0
    A2notA1 = 0
    bothIncorrect = 0
        
    for p1 in correctPointsA1:
        if p1 in correctPointsA2:
            bothCorrect+=1
        else:
            A1notA2 += 1
    for p2 in correctPointsA2:
        if p2 not in correctPointsA1:
            A2notA1 += 1
    for p1 in incorrectPointsA1:
        if p1 in incorrectPointsA2:
            bothIncorrect += 1
            
    print 
    print 
    print '----------------------------------'
    print 
    print 'Correct by both:    '+str(bothCorrect)
    print 'Correct by Tree1:   '+str(A1notA2)
    print 'Correct by Tree2:   '+str(A2notA1)
    print 'Correct by neither: '+str(bothIncorrect)
    
def pruneToDepth(tree, depth):
    if tree == None:
        return None
    if depth == 0:
        allLabelFreqs = {}
        for p in tree.dataPoints:
            if p.location not in allLabelFreqs:
                allLabelFreqs[p.location] = 0
            allLabelFreqs[p.location] += 1
        bestLabel = None
        highFreq = 0
        for l in allLabelFreqs:
            if allLabelFreqs[l] > highFreq:
                highFreq = allLabelFreqs[l]
                bestLabel = l
        newTree = Node(None, None, tree.dataPoints, attr=tree.attr, value=tree.value, label=bestLabel)
        return newTree
    newTree = Node(None, None, tree.dataPoints, attr=tree.attr, value=tree.value, label=tree.label)
    newTree.left = pruneToDepth(tree.left, depth - 1)
    newTree.right = pruneToDepth(tree.right, depth - 1)
    return newTree
    
def genKSplits(dataset, k):
    splitSize = math.ceil(float(len(dataset))/k)
    
    splits = {}
    
    tot = len(dataset)
    curSplit = 0
    for i in range(0, tot):
        curSplit = math.floor(float(i)/splitSize)
        randomInd = random.randint(0, len(dataset)-1)
        randomP = dataset[randomInd]
        
        if curSplit not in splits:
            splits[curSplit] = []
        splits[curSplit].append(randomP)
        del dataset[randomInd]
    return splits.values()
    
def findOptimalDepthTree(training, validation):
    mainTree = buildTree(training)
    
    besterror = sys.maxint
    bestdepth = -1
    for i in range(0, mainTree.getDepth()):
        newTree = pruneToDepth(mainTree, i)
        error = classifyDataSet(newTree, validation, out=False)
        
        if error < besterror:
            bestdepth = i
            besterror = error
    
    return pruneToDepth(mainTree, bestdepth)
    
dataPoints = loadData('wifi.train')
testData = loadData('wifi.test')

tree = buildTree(dataPoints)

print
print
print "|||||||  Full Tree -- Training  |||||||"
classifyDataSet(tree, dataPoints)
print
print
print "|||||||  Full Tree -- Testing  |||||||"
classifyDataSet(tree, testData)

prunedTree = buildTree(dataPoints)
prunedTree = pruneToDepth(prunedTree, 2)
print
print
print "|||||||  Depth-2 Tree -- Training  |||||||"
classifyDataSet(prunedTree, dataPoints)
print
print
print "|||||||  Depth-2 Tree -- Testing  |||||||"
classifyDataSet(prunedTree, testData)

compareAlgorithms(tree, prunedTree, testData)

print 
print "________________________________________________________"
print
print " Cross validation"
print "________________________________________________________"
print 

if False:
    combinedSample = loadData('wifi.crossval')
    
    k = 10  #
    
    crossValSplits = genKSplits(combinedSample, k)
    
    for i in range(0, k):
        testset = []
        trainingset = []
        
        for j in range(0, k):
            if j == i:
                testset.extend(crossValSplits[j])
            else:
                trainingset.extend(crossValSplits[j])
        
        tree = buildTree(trainingset)
        prunedtree = pruneToDepth(buildTree(trainingset), 2)
        
        print 'Split '+str(i)
        print 'full tree:  '+str(classifyDataSet(tree, testset, out=False))
        print 'depth 2:    '+str(classifyDataSet(prunedtree, testset, out=False))
        compareAlgorithms(tree, prunedtree, testset)

print 
print "________________________________________________________"
print
print " Validation Set Testing"
print "________________________________________________________"
print

if False:
    combinedSample = loadData('wifi.crossval')
    
    k = 10
    crossValSplits = genKSplits(combinedSample, k)

    # Do 10-fold cross val
    for i in range(0, k):
        print " ----- Cross Fold [ "+str(i)+" ] -------"
        testset = []
        trainingset = []
        
        for j in range(0, k):
            if j == i:
                testset.extend(crossValSplits[j])
            else:
                trainingset.extend(crossValSplits[j])
        
        #Divide up validation sets
        for vv in range(1, 10):
            vRatio = float(vv)/10
            finaltrainset = []
            validationset = []
            
            random.shuffle(trainingset)
            
            j = 0
            for p in trainingset:
                if j > (vRatio * len(trainingset)):
                    finaltrainset.append(p)
                else:
                    validationset.append(p)
                    
                j += 1
                
            tree = findOptimalDepthTree(finaltrainset, validationset)
            
            print " Val ratio: "+str(vRatio)+" |     Optimal Depth: "+str(tree.getDepth())+"  Val Error: "+str(classifyDataSet(tree, validationset, out=False))+" Test Error: "+str(classifyDataSet(tree, testset, out=False))
            
if False:            
    trainData = loadData('wifi.train')
    test_data = loadData('wifi.test')
    validate_error = {}
    test_error = {}
    
    #Repeat for (ratio) in [0.1:0.9]
    for vv in range(1, 10):
        print " ----  Ratio "+str(float(vv)/10)
        
        vRatio = float(vv)/10
        validate_error[vRatio] = []
        test_error[vRatio] = []
        
    #~~ Repeat 10 times:
        for cr in range(0, 10):
            
    #~~~~~~~0. Shuffle training set
            random.shuffle(trainData)
    
    #~~~~~~~1. Split your training set into (new_train) and (validate) using that (ratio) 
            new_train = []
            validate = []
            for i, p in enumerate(trainData):
                if i > (vRatio * len(trainData)):
                    new_train.append(p)
                else:
                    validate.append(p)
            
    #~~~~~~~ 2. Train your tree on (new_train)
    #~~~~~~~3. Until (validate_error) is the smallest (iterate through multiple tree depths)
            bestTree = findOptimalDepthTree(new_train, validate)
    
    #~~~~~~~ 4. Record lowest validation error: validate_error[ratio].append(lowest_validate_error)
            er = classifyDataSet(bestTree, validate, out=False)
            print "  Validate Error:  "+str(er)
            validate_error[vRatio].append(er)
    
    #~~~~~~~ 5. Test tree with the lowest validation error on (test_set), record it: test_error[ratio].append(test_error)
            er = classifyDataSet(bestTree, test_data, out=False)
            print "  Test Error    :  "+str(er)
            test_error[vRatio].append(er)
            
    print "validate_error"
    print ",1,2,3,4,5,6,7,8,9,10"
    for ratio in validate_error:
        sys.stdout.write(str(ratio))
        for err in validate_error[ratio]:
            sys.stdout.write(','+str(err));
        sys.stdout.write('\n')
        
    print "test_error"
    print ",1,2,3,4,5,6,7,8,9,10"
    for ratio in test_error:
        sys.stdout.write(str(ratio))
        for err in test_error[ratio]:
            sys.stdout.write(','+str(err));
        sys.stdout.write('\n')
    
    print 
