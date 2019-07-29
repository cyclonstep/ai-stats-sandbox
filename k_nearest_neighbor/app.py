import pandas as pd
import numpy as np
import math
import operator
from pprint import pprint

### Start of Step 1

# importing data
data = pd.read_csv("iris.csv")

### End of Step 1

#data.head()

# Defining a function which calculates euclidian distance between two data points
def euclideanDistance(data1, data2, length):
    distance = 0 
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)

# Defining our KNN model
def knn(trainingSet, testInstance, k):
    distances = {}
    sort = {}

    length = testInstance.shape[1]

    ### Start of Step 3

    # Calculating euclidean distance between each row of training data and test data
    for x in range(len(trainingSet)):
        #### Start of Step 3.1
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)

        distances[x] = dist[0]
        #### End of Step 3.1
    
    #print("distances:")
    #pprint(distances)


    #### Start of Step 3.2
    # Sorting them on the basis of distance
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
    #### End of Step 3.2

    #print("sorted distances: ")
    #pprint(sorted_d)

    neighbors = []

    #### Start of Step 3.3
    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    #### End of Step 3.3

    #print("top {} distances: ".format(k))
    #pprint(neighbors)

    classVotes = {}

    #### Start of Step 3.4
    # Calculating the most freq class in the neighbors 
    #pprint(trainingSet)
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]

        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    #### End of Step 3.4


    #### Start of Step 3.5
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

    return(sortedVotes[0][0], neighbors)

    #### End of Step 3.5

# Creating a dummy testset
testSet = [[1.2, 1.6, 0.1, 0.5]]
test = pd.DataFrame(testSet)

#print(test)

### Start of Step 2
# Setting number of neighbors = 1
k = 3
### End of Step 2
# Running KNN model 
result, neigh = knn(data, test, k)

# Predicted class
print(result)

# Nearest  neighbor
print(neigh)
