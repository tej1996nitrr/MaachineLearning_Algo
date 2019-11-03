#%%
import csv
#opening file in text mode
with open('Data\iris.data', 'rt') as csvfile:
	lines = csv.reader(csvfile)
	for row in lines:
		print( ', '.join(row))
#%%
import csv
import random
def loadDataset(filename,split,trainingSet=[],testSet=[]):
    with open(filename,'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset  = list(lines)
        for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])


#%%

trainingSet=[]
testSet=[]
loadDataset('Data\iris.data', 0.66, trainingSet, testSet)
print( 'Train: ' + repr(len(trainingSet)))
print ('Test: ' + repr(len(testSet)))

#%%	
import math
def euclideanDistance(first, second, length):
	distance = 0
	for x in range(length):
		distance += pow((first[x] - second[x]), 2)
	return math.sqrt(distance)
# data1 = [2, 2, 2, 'a']
# data2 = [4, 4, 4, 'b']
# distance = euclideanDistance(data1, data2, 3)
# print ('Distance: ' + repr(distance))


#%%
import operator
def get_neighbours(trainingSet,testInstance,k):
	distances=[]
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance,trainingSet[x],length)
		distances.append((trainingSet[x],dist))
	distances.sort(key=operator.itemgetter(1))
	neighbours=[]
	for x in range(k):
		neighbours.append(distances[x][0])
	return neighbours

# trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
# testInstance = [5, 5, 5]
# k = 1
# neighbors = get_neighbours(trainSet, testInstance, 1)
# print(neighbors)

#%%
import operator
def getPrediction(neigh):
	classvotes={}
	for x in range(len(neigh)):
		prediction = neigh[x][-1]
		if prediction in classvotes:
			classvotes[prediction]+=1
		else:
			classvotes[prediction]=1
	sortedVotes = sorted(classvotes.items(),key=operator.itemgetter(1),reverse=True)
	return sortedVotes[0][0]

	
# neighbors = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
# response = getPrediction(neighbors)
# print(response)
#%%
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0


#%%
def main():
	trainingSet=[]
	testSet = []
	split  = 0.66
	loadDataset('Data\iris.data',split,trainingSet,testSet)
	print('Train set: ' + repr(len(trainingSet)))
	print('Test set: ' + repr(len(testSet)))
	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = get_neighbours(trainingSet, testSet[x], k)
		result = getPrediction(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
main()


#%%
