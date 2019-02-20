from data_utils import load_dataset
import math
import copy
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from operator import itemgetter

x_train, x_valid, x_test, y_train, y_valid, y_test = ([[],[],[],[],[]], [[],[],[],[],[]],
                                                     [[],[],[],[],[]], [[],[],[],[],[]],
                                                     [[],[],[],[],[]], [[],[],[],[],[]])
x_train[0], x_valid[0], x_test[0], y_train[0], y_valid[0], y_test[0] = list(load_dataset('mauna_loa'))
x_train[1], x_valid[1], x_test[1], y_train[1], y_valid[1], y_test[1] = list(load_dataset('rosenbrock', n_train=5000, d=2))
x_train[2], x_valid[2], x_test[2], y_train[2], y_valid[2], y_test[2] = list(load_dataset('pumadyn32nm'))
x_train[3], x_valid[3], x_test[3], y_train[3], y_valid[3], y_test[3] = list(load_dataset('iris'))
x_train[4], x_valid[4], x_test[4], y_train[4], y_valid[4], y_test[4] = list(load_dataset('mnist_small'))

Inf = 100000
###############################################################################
###############################Helper Functions:###############################
###############################################################################
def RMSE(measurements, actuals):
    n = len(measurements)
    sum = 0
    for i in range(n):
        sum += (measurements[i] - actuals[i])**2
    return math.sqrt(sum/n)
        
def numberCorrect(measurements, actuals):
    n = len(measurements)
    corrects = 0
    for i in range(n):
        if actuals[i][measurements[i]] == True:
            corrects += 1
    return corrects

def l1(a, b):
    return abs(a - b)

def l2(a, b):
    if type(a) == np.float64 or type(a) == float:
        return l1(a, b)
    dimension = len(a)
    inner = 0
    for i in range(dimension):
        inner += (a[i] - b[i])**2
    return math.sqrt(inner)

def lInf(a, b):
    if type(a) == np.float64 or type(a) == float:
        return l1(a, b)
    dimension = len(a)
    maxVal = 0
    for i in range(dimension):
        if abs(a[i] - b[i]) > maxVal:
            maxVal = abs(a[i] - b[i])
    return maxVal

def generateFolds(data):
    size = len(data)
    foldSet = copy.deepcopy(list(data))

    f1 = [foldSet[math.ceil(size/5):], foldSet[:math.ceil(size/5)]]
    
    t2 = foldSet[:math.ceil(size/5)]
    t2.extend(foldSet[math.ceil(size/5 * 2):])
    f2 = [t2, foldSet[math.ceil(size/5):math.ceil(size/5*2)]]                                           
                                         
    t3 = foldSet[:math.ceil(size/5*2)]
    t3.extend(foldSet[math.ceil(size/5*3):])
    f3 = [t3, foldSet[math.ceil(size/5*2):math.ceil(size/5*3)]]                                             
    
    t4 = foldSet[:math.ceil(size/5*3)]
    t4.extend(foldSet[math.ceil(size/5*4):])
    f4 = [t4, foldSet[math.ceil(size/5*3):math.ceil(size/5*4)]]                                            
    
    f5 = [foldSet[:math.ceil(size/5*4)], foldSet[math.ceil(size/5*4):]]
    
    return [f1, f2, f3, f4, f5]
###############################################################################
################################Question 1:####################################
##########################Cross-Validation K-NN:###############################
def kNNCV(xData, yData, k, distanceMetric):
    tupleList = list(zip(xData, yData))
    random.shuffle(tupleList)
    xData, yData = zip(*tupleList)
    xData = list(xData)
    yData = list(yData)
    xFolds = generateFolds(xData)
    yFolds = generateFolds(yData)
    errors = []
    predictionSet = []
    yTestFolds = []
    for i in range(5):
        xTrain = xFolds[i][0]
        yTrain = yFolds[i][0]
        xTest = xFolds[i][1]
        yTest = yFolds[i][1]
        predictions = []
        error, predictions, yTest = kNNG(xTrain, yTrain, xTest, yTest, k, distanceMetric)
        errors.append(error)
        predictionSet.append(predictions)
        yTestFolds.append(yTest)
    averageError = 0
    for i in range(len(errors)):
        averageError += errors[i]
    averageError = averageError/len(errors)
    return averageError, predictionSet, yTestFolds

################################Question 2:####################################
########################KNN for Classification - CV:###########################
def classkNNCV(xData, yData, k, distanceMetric):
    tupleList = list(zip(xData, yData))
    random.shuffle(tupleList)
    xData, yData = zip(*tupleList)
    xData = list(xData)
    yData = list(yData)
    xFolds = generateFolds(xData)
    yFolds = generateFolds(yData)
    corrects = []
    for i in range(5):
        xTrain = xFolds[i][0]
        yTrain = yFolds[i][0]
        xTest = xFolds[i][1]
        yTest = yFolds[i][1]
        predictions = []
        for j in range(len(xTest)):
            if distanceMetric == "l1":
                lenArray = list(np.sum(abs(xTrain-xTest[j]),axis=1))
            elif distanceMetric == "l2":
                lenArray = list(np.sqrt(np.sum(np.square(xTrain-xTest[j]),axis=1)))
            elif distanceMetric == "lInf":
                lenArray = list(np.max(abs(xTrain-xTest[j]),axis=1))
            else:
                print("Invalid Distance Metric")
                return 0
            sortedLenArray = sorted(lenArray)
            minValues = sortedLenArray[:k]
            majorityArray = list([ 0 for i in range(len(yTrain[0]))])
            for m in range(k):
                minIndex = lenArray.index(minValues[m])
                majorityArray[list(yTrain[minIndex]).index(True)] += 1
            predictions.append(majorityArray.index(max(majorityArray)))
        corrects.append([numberCorrect(predictions, yTest), len(xTest)])
    return corrects
##########################KNN for Classification:##############################
def classkNN(xTrain, yTrain, xTest, yTest, k, distanceMetric):
    tupleList = list(zip(xTrain, yTrain))
    random.shuffle(tupleList)
    xTrain, yTrain = zip(*tupleList)
    xTrain = list(xTrain)
    yTrain = list(yTrain)
    predictions = []
    for j in range(len(xTest)):
        if distanceMetric == "l1":
            lenArray = list(np.sum(abs(xTrain-xTest[j]),axis=1))
        elif distanceMetric == "l2":
            lenArray = list(np.sqrt(np.sum(np.square(xTrain-xTest[j]),axis=1)))
        elif distanceMetric == "lInf":
            lenArray = list(np.max(abs(xTrain-xTest[j]),axis=1))
        else:
            print("Invalid Distance Metric")
            return 0
        sortedLenArray = sorted(lenArray)
        minValues = sortedLenArray[:k]
        majorityArray = list([ 0 for i in range(len(yTrain[0]))])
        for m in range(k):
            minIndex = lenArray.index(minValues[m])
            majorityArray[list(yTrain[minIndex]).index(True)] += 1
        predictions.append(majorityArray.index(max(majorityArray)))
    corrects = [numberCorrect(predictions, yTest), len(xTest)]
    return corrects
################################Question 3:####################################
##########################Types of KNN Implementations:########################
########################Normal KNN with Double-For Loop:#######################
def kNNA(xTrain, yTrain, xTest, yTest, k, distanceMetric):
    xTrain = list(xTrain)
    yTrain = list(yTrain)
    xTest = list(xTest)
    yTest = list(yTest)  
    predictions = []
    for j in range(len(xTest)):
        lenArray = []
        for x in range(len(xTrain)):
            if distanceMetric == "l1":
                lenArray.append([l1(xTest[j][0], xTrain[x][0]), yTrain[x]])
            elif distanceMetric == "l2":
                lenArray.append([l2(xTest[j][0], xTrain[x][0]), yTrain[x]])
            elif distanceMetric == "lInf":
                lenArray.append([lInf(xTest[j][0], xTrain[x][0]), yTrain[x]])
            else:
                print("Invalid Distance Metric")
                return 0
        lenArray = sorted(lenArray, key = itemgetter(0))
        prediction = 0
        for m in range(k):
            prediction += lenArray[m][1]
        prediction = prediction/k
        predictions.append(prediction)
    error = RMSE(predictions, yTest)
    return error, predictions, yTest

#########################Partially Vectorized KNN:#############################
def kNNB(xTrain, yTrain, xTest, yTest, k):
    xTrain = list(xTrain)
    yTrain = list(yTrain)
    xTest = list(xTest)
    yTest = list(yTest)  
    predictions = []
    for j in range(len(xTest)):
        lenArray = list(np.array(np.sqrt(np.sum(np.square(xTrain-xTest[j]),axis=1))))
        sortedLenArray = sorted(lenArray)
        minKValues = sortedLenArray[:k]
        prediction = 0
        for m in range(k):
            minIndex = lenArray.index(minKValues[m])
            prediction += yTrain[minIndex]
        prediction = prediction/k
        predictions.append(prediction)
        
    error = RMSE(predictions, yTest)
    return error, predictions, yTest

##############################Fully Vectorized KNN:############################
def kNNC(xTrain, yTrain, xTest, yTest, k):
    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)
    xTest = np.array(xTest)
    yTest = np.array(yTest)
    xTest = xTest[:, None]
    
    targets = np.array([yTrain]*len(xTest))
    targets = np.squeeze(targets)
    
    diff = xTrain - xTest
    distances = np.square(diff)
    distances = np.sum(distances, axis=2)
    distances = np.sqrt(distances)

    struct = np.zeros((len(distances),len(distances[0])), dtype = [('distances', np.float64),('targets',np.float64)])
   
    struct['distances'] = distances 
    struct['targets'] = targets
    
    ascStruct = np.sort(struct, axis = 1, order = ('distances'))

    minKCombination = ascStruct[:,:k]
    minKTargets = minKCombination['targets']
    predictions = np.sum(minKTargets, axis=1)/k
    error = RMSE(predictions, yTest)
    return error, predictions, yTest

#############################k-d Tree Implementaion:###########################
def kNND(xTrain, yTrain, xTest, yTest, k, distanceMetric):
    tree = KDTree(xTrain)
    dist, ind = tree.query(xTest, k=k)
    predictions = np.mean(yTrain[ind].T, axis=1)[0]
    error = RMSE(predictions, yTest)
    return error, predictions, yTest

###############################################################################
def kNNG(xTrain, yTrain, xTest, yTest, k, distanceMetric):
    xTrain = list(xTrain)
    yTrain = list(yTrain)
    xTest = list(xTest)
    yTest = list(yTest)  
    predictions = []
    for j in range(len(xTest)):
        if distanceMetric == "l1":
            lenArray = np.sum(abs(xTrain-xTest[j]),axis=1)
        elif distanceMetric == "l2":
            lenArray = np.sqrt(np.sum(np.square(xTrain-xTest[j]),axis=1))
        elif distanceMetric == "lInf":
            lenArray = list(np.max(abs(xTrain-xTest[j]),axis=1))
        else:
            print("Invalid Distance Metric")
            return 0
        sortedLenArray = sorted(lenArray)
        minKValues = sortedLenArray[:k]
        prediction = 0
        for m in range(k):
            minIndex = list(lenArray).index(minKValues[m])
            prediction += yTrain[minIndex]
        prediction = prediction/k
        predictions.append(prediction)
        
    error = RMSE(predictions, yTest)
    return error, predictions, yTest
###############################################################################
######################SVD Regression/Classification:###########################
def SVDReg(xTrain, yTrain, xTest, yTest, xValid, yValid, C):
    xMerged = np.vstack([xValid, xTrain])
    yMerged = np.vstack([yValid, yTrain])
    xTrain = np.insert(xMerged, 0, 1, axis = 1)
    xTest = np.insert(xTest, 0, 1, axis = 1)
    u, s, vh = np.linalg.svd(xTrain, full_matrices=False)
    s = np.diag(s)
    w = np.matmul(np.transpose(vh), np.matmul(np.linalg.inv(s), np.matmul(np.transpose(u), yMerged)))
    predictions = np.matmul(xTest, w)
    if C == True:
        predictionIndices = []
        for i in range(len(predictions)):
            predictions[i] = abs(predictions[i])
            predictionIndices.append(list(predictions[i]).index(max((predictions[i]))))
        corrects = [numberCorrect(predictionIndices, yTest), len(xTest)]
        return corrects
    else:
        error = RMSE(predictions, yTest)
        return error, predictions, yTest

###########################Functions for Plotting:#############################
def CVLvsK(kMax, dataInd, distanceMetric):
    optimal = 0
    kArray = [ k for k in range(1, kMax + 1)]
    errorArray = []
    for k in range(1, kMax + 1):
        averageError, predictionSet, yTestFolds = kNNCV(x_train[dataInd], y_train[dataInd], k, distanceMetric)
        errorArray.append(averageError)
        print("k =", k, "error =", averageError)
        if k == 1 or averageError < optimal:
            optimal = averageError
            optimalK = k 
    print("Optimal K =", optimalK)
    print("Cross-Validation RMSE:", optimal)
    fig, ax = plt.subplots()
    ax.plot(kArray, errorArray, 'b', linewidth=2.5)
    plt.title("Cross Validation Loss vs K")
    plt.xlabel("K")
    plt.ylabel("Loss")
    plt.show()
    fig.savefig("CrossValidationLoss-vs-K.png")
    return
    
def CVPredCurves():
    fig, ax = plt.subplots()
    for k in range(1, 6):
        averageError, predictionSet, yTestFolds = kNNCV(x_train[0], y_train[0], k, "l2")
        flattenedPredictionSet = [value for sublist in predictionSet for value in sublist]
        flattenedyTestFolds = [value for sublist in yTestFolds for value in sublist]
        if k == 1:
            ax.plot(flattenedyTestFolds, flattenedPredictionSet, 'bo', markersize = 2, label = "k = {}".format(k))
        elif k == 2:
            ax.plot(flattenedyTestFolds, flattenedPredictionSet, 'ro', markersize = 2, label = "k = {}".format(k))
        elif k == 3:
            ax.plot(flattenedyTestFolds, flattenedPredictionSet, 'co', markersize = 2, label = "k = {}".format(k))
        elif k == 4:
            ax.plot(flattenedyTestFolds, flattenedPredictionSet, 'yo', markersize = 2, label = "k = {}".format(k))
        else:
            ax.plot(flattenedyTestFolds, flattenedPredictionSet, 'mo', markersize = 2, label = "k = {}".format(k))
        
    ax.legend()
    plt.title('Cross-Validation Prediction Curves')
    plt.xlabel('Actuals')
    plt.ylabel('Predictions')
    plt.show()
    fig.savefig("CrossValidationPredictionCurves.png")
    return

def PredvsTest():
    error, predictions, yTest = kNNA(x_train[0], y_train[0], x_test[0], y_test[0], 2, "l2")
    fig, ax = plt.subplots()
    ax.plot(yTest, predictions, 'ms', markersize = 0.8)
    plt.title("Test Data Predictions")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()
    fig.savefig("TestDataPredictions.png")
    return
###############################################################################
def question1():
    CVPredCurves()
    CVLvsK(30, 0, "l1")
    PredvsTest()
    error, predictions, yTest = kNNCV(x_train[0], y_train[0], 2, "l1")
    print("Training RMSE Optimal mauna_loa:", error)
    error, predictions, yTest = kNNG(x_train[0], y_train[0], x_test[0], y_test[0], 2, "l1")
    print("Test RMSE Optimal mauna_loa:", error)
    error, predictions, yTest = kNNCV(x_train[1], y_train[1], 1, "lInf")
    print("Training RMSE Optimal rosenbrock:", error)
    error, predictions, yTest = kNNG(x_train[1], y_train[1], x_test[1], y_test[1], 1, "lInf")
    print("Test RMSE Optimal rosenbrock:", error)
    error, predictions, yTest = kNNCV(x_train[2], y_train[2], 4, "lInf")
    print("Training RMSE Optimal pumadyn32nm:", error)
    error, predictions, yTest = kNNG(x_train[2], y_train[2], x_test[2], y_test[2], 16, "l1")
    print("Testing RMSE Optimal pumadyn32nm:", error)
    
def question2():
    corrects = classkNN(x_train[3], y_train[3], x_test[3], y_test[3], 7, "l1")
    print("Percentage of Correct Predictions, iris:", corrects[0]/corrects[1])
    corrects = classkNN(x_train[4], y_train[4], x_test[4], y_test[4], 3, "l1")
    print("Percentage of Correct Predictions, mnist_small:", corrects[0]/corrects[1])
    for j in range(3, 5):
        if j == 3:
            print("Iris Dataset:")
        else:
            print("Mnist Small Dataset:")
        for k in range(1, 36): 
            c1 = classkNNCV(x_train[j], y_train[j], k, "l1")
            c2 = classkNNCV(x_train[j], y_train[j], k, "l2")
            cInf = classkNNCV(x_train[j], y_train[j], k, "lInf")
            c1Temp = [0,0]
            c2Temp = [0,0]
            cInfTemp = [0,0]
            for i in range(5):
                c1Temp[0] += c1[i][0]
                c1Temp[1] += c1[i][1]
                c2Temp[0] += c2[i][0]
                c2Temp[1] += c2[i][1]
                cInfTemp[0] += cInf[i][0]
                cInfTemp[1] += cInf[i][1]
            c1 = c1Temp[0]/c1Temp[1]
            c2 = c2Temp[0]/c1Temp[1]
            cInf = cInfTemp[0]/cInfTemp[1]
            if k == 1:
                maxL1, maxL1K = c1, k
                maxL2, maxL2K = c2, k
                maxLInf, maxLInfK = cInf, k
                
            if c1 > maxL1:
                maxL1, maxL1K = c1, k
            if c2 > maxL2:
                maxL2, maxL2K = c2, k
            if cInf > maxLInf:
                maxLInf, maxLInfK = cInf, k
                
            print("k = {},".format(k), "Average Correct L1 =", c1)
            print("k = {},".format(k), "Average Correct L2 =", c2)
            print("k = {},".format(k), "Average Correct LInf =", cInf)
        if max(maxL1, maxL2, maxLInf) == maxL1:
            print("Optimal Distance Metric = L1")
            print("Optimal K: ", maxL1K)
        elif max(maxL1, maxL2, maxLInf) == maxL2:
            print("Optimal Distance Metric = L2")
            print("Optimal K: ", maxL2K)
        else:
            print("Optimal Distance Metric = LInf")
            print("Optimal K: ", maxLInfK)
    return
    
def question3():
    fig, ax = plt.subplots()
    dValues = [m for m in range(2, 11)]
    kNNATimes = []
    kNNBTimes = []
    kNNCTimes = []
    kNNDTimes = []
    for m in range(2, 11):
        print("d =", m, ":")
        x_train[1], x_valid[1], x_test[1], y_train[1], y_valid[1], y_test[1] = list(load_dataset('rosenbrock', n_train=5000, d=m))
        startTime = time.time()
        error, predictions, yTest = kNNA(x_train[1], y_train[1], x_test[1], y_test[1], 5, "l2")
        elapsedA = time.time() - startTime
        kNNATimes.append(elapsedA)
        print("Time elapsed for normal KNN:", elapsedA)
        startTime = time.time()
        error, predictions, yTest = kNNB(x_train[1], y_train[1], x_test[1], y_test[1], 2)
        elapsedB = time.time() - startTime
        kNNBTimes.append(elapsedB)
        print("Time elapsed for partially vectorized KNN:", elapsedB)
        startTime = time.time()
        error, predictions, yTest = kNNC(x_train[1], y_train[1], x_test[1], y_test[1], 5)
        elapsedC = time.time() - startTime
        kNNCTimes.append(elapsedC)
        print("Time elapsed for fully vectorized KNN:", elapsedC)
        startTime = time.time()
        error, predictions, yTest = kNND(x_train[1], y_train[1], x_test[1], y_test[1], 5, "l2")
        elapsedD = time.time() - startTime
        kNNDTimes.append(elapsedD)
        print("Time elapsed for k-d tree structure:", elapsedD)
    ax.plot(dValues, kNNATimes, marker = '.', mew = 3, color = 'm', label = 'Normal k-NN')
    ax.plot(dValues, kNNBTimes, marker = '.', mew = 3, color = 'b', label = 'Partially Vectorized k-NN')
    ax.plot(dValues, kNNCTimes, marker = '.', mew = 3, color = 'c', label = 'Vectorized k-NN')
    ax.plot(dValues, kNNDTimes, marker = '.', mew = 3, color = 'y', label = 'k-d Tree Implementation')
    ax.legend()
    plt.title("Time Elapsed for Various Algorithms")
    plt.xlabel("d Values")
    plt.ylabel("Time")
    fig.savefig("kNNTimes-vs-dValues.png")
    plt.show()
    
def question4():
    for i in range(3):
        if i == 1:
            startTime = time.time()
            error, predictions, yTest = SVDReg(x_train[i], y_train[i], x_test[i], y_test[i], x_valid[i], y_valid[i], False)
            timeElapsed = time.time() - startTime
            print("Error for the {}th dataset:".format(i), error)
            print("Time elapsed for rosenbrock:", timeElapsed)
        else:
            error, predictions, yTest = SVDReg(x_train[i], y_train[i], x_test[i], y_test[i], x_valid[i], y_valid[i], False)
            print("Error for the {}th dataset:".format(i), error)
    for i in range(3,5):
        corrects = SVDReg(x_train[i], y_train[i], x_test[i], y_test[i], x_valid[i], y_valid[i], True)
        print("Accuracy for the {}th dataset:".format(i), corrects[0]/corrects[1])
    
if __name__ == "__main__":
    #question1()
    #question2()
    #question3()
    question4()
     
    