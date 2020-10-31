
import re
import numpy as np
import math

from prettytable import PrettyTable

testFile = "Test.txt"
trainFile = "Train.txt"


def summary(numbers):
    avg = sum(numbers)/float(len(numbers))
    var = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)) #sample -1 
    return avg, var

    
def normalProb(x, mean, var):
    if var==0:
        return int(x == mean)
    exponent = math.exp( - (x-mean)**2 / (2*var) )
    return (1/(math.sqrt(2* math.pi * var))) * exponent



print('------------------------------------Naive Bayes Classifier----------------------')
#Reading training set
dataset = []
numFeature, numClass, datasetLen = 0, 0, 0

with open(trainFile, 'r') as file:
    frst = list(filter(None, re.split(r'[-:;,\s]\s*', file.readline())))

    numFeature, numClass, datasetLen = int(frst[0]), int(frst[1]), int(frst[2])

    for line in file:
        if not line.strip():
            continue
        fields = list(filter(None, re.split(r'[-:;,\s]\s*', line))) 
        dataset.append([float(x) for x in fields]) #or int




data_mat = np.array(dataset)

classSet = set(data_mat[:,-1])
feature_mat = []
classFreq = []
for i in classSet:
    feature_mat.append(data_mat[data_mat[:, numFeature] == i, :])
    print("Class no-",i)
    # print(feature_mat)
    print("Count = ", len(feature_mat[-1]))
    classFreq.append(len(feature_mat[-1]))
print(classFreq)
classFreq = np.array(classFreq) / datasetLen * 100.0
print(classFreq)

#contains mean and variance of each class 
# first row of 2-d array is mean of individual features 
#second one is variance 
smry = []
for i in range(len(classSet)):
    smry.append( np.apply_along_axis(summary, 0, feature_mat[i]) )


#reading test sets
dataset = []

with open(testFile, 'r') as file:
    for line in file:
        if not line.strip():
            continue
        fields = list(filter(None, re.split(r'[-:;,\s]\s*', line))) 
        dataset.append([float(x) for x in fields]) #or int

mismatchCnt = 0
sample_id = 0
t = PrettyTable(['#Sample', 'Features', 'Actual_Class', 'Prediceted_Class'])

for test in dataset:
    predictClass = 1
    bestProb = -1

    for i in range(len(classSet)):

        prob = classFreq[i]
        for j in range(numFeature):
            prob*= normalProb( test[j], smry[i][0][j], smry[i][1][j])
        if prob>bestProb:
            bestProb = prob
            predictClass = i

    predictClass = list(classSet)[predictClass]
    if(test[-1] != predictClass):
        mismatchCnt = mismatchCnt + 1
        t.add_row([sample_id,test[:-1],test[-1],predictClass])

    sample_id = sample_id + 1

print(t)
accuracy = (len(dataset)- mismatchCnt) * 100.0 /len(dataset)
print("Accuracy = " ,  accuracy , "%" )








