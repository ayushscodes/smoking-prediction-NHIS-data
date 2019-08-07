import numpy as np
from read_data import *
from scipy.spatial.distance import cosine
from scipy import stats
from math import *

np.set_printoptions(threshold=10000)


def cos_dist(X1, X2):
    mode1 = 0
    mode2 = 0 
    prod = 0
    for i in range( len(X1) ):
        if X1[i]==99.99 or X2[i]==99.99:
            continue
        else:
            mode1 += X1[i]**2
            mode2 += X2[i]**2
            prod += X1[i] * X2[i]
    
    return 1.0 - prod/(sqrt(mode1) * sqrt(mode2))


def jaccard_dist(X1, X2):
    union = 0.0
    intersection = 0.0
    for i in range( len(X1) ):
        if X1[i]==99.99 or X2[i]==99.99:
            continue
        elif X1[i] == X2[i]:
            intersection += 1
            union += 1
        else:
            union += 2
        
    return 1 - intersection / union


def classify(X, X_train, Y_train):
    index = 0
    min_dist = 1
    for i in range(len(X_train)):
        #dist = cos_dist(X_train[i],X)
        dist = jaccard_dist(X_train[i],X)
        if dist < min_dist:
            index = i 
            min_dist = dist
            
    return Y_train[index]


def multi_classify(X, X_train, Y_train):
    indices = []
    min_dist = 1
    for i in range(len(X_train)):
        #dist = cos_dist(X_train[i], X)
        dist = jaccard_dist(X_train[i],X)
        indices.append((i, dist))

    indices = sorted(indices, key = lambda x: x[1])
    k = 29
    top_k = indices[:k]
    ys = []
    for i in range(len(top_k)):
        index, dist = top_k[i]
        ys.append(Y_train[index])
    return stats.mode(ys)[0][0]


def test(X_train, Y_train, X_test, Y_test):
    error = 0
    for i in range(len(X_test)):
        if multi_classify(X_test[i], X_train, Y_train) != Y_test[i]:
            error +=1
            
    return 1 - error*1.0 / len(X_test)
    


X_train, Y_train, X_val, Y_val, X_test, Y_test, headers = get_split_data()

print(test(X_train, Y_train, X_test, Y_test))
