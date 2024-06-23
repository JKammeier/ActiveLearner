# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:59:31 2024

@author: JannisKammeier
"""

# %% imports
from tensorflow.keras import datasets, losses, models, layers
# import matplotlib.pyplot as plt
from numpy import ndarray, append, concatenate, log2, uint8
from random import shuffle
from sys import stdout
# from joblib import Parallel, delayed
from heapq import nlargest

# tf.config.list_physical_devices()
# %% 
class ActiveLearner:
    def __init__(self, numInitDatapoints:int=100, verbose:int=0, processes:int=-1) -> None:
        (xTrain, yTrain), (self.xTest, self.yTest) = datasets.mnist.load_data()  # load mnist dataset
        
        # shuffle training data
        randomOrder = self.createRandomOrder(length=len(yTrain))
        xTrain = xTrain[randomOrder]
        yTrain = yTrain[randomOrder]
        
        # when a negative Number is given, use all of the available training data
        if numInitDatapoints <= 0:
                numInitDatapoints = len(yTrain)
        
        self.xTrainLabeled:ndarray[ndarray[ndarray[uint8]]] = xTrain[:numInitDatapoints]    # split training data into "labeled" and "unlabeled" for active learning
        self.yTrainLabeled:ndarray[ndarray[uint8]] = yTrain[:numInitDatapoints]
        self.xTrainUnlabeled:ndarray[ndarray[ndarray[uint8]]] = xTrain[numInitDatapoints:]
        self.yTrainUnlabeled:ndarray[ndarray[uint8]] = yTrain[numInitDatapoints:]
        
        self.completedIterations:int = -1   # shows how many iterations the active learner has gone through (-1=model has not been trained, 0 = model has been trianed but no active learning iteration has been made)
        self.numberLabels:int = []
        self.losses:float = []
        self.accuracies:float = []
        self.verbose:int = verbose
        self.processes:int = processes
        
        self.buildModel()
        self.activeLearningLoop(0)
    
    
    
    def buildModel(self) -> None:
        self.trainingModel = models.Sequential([
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(100, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(10)
            ])
        
        # loss and optimizer
        lossFct = losses.SparseCategoricalCrossentropy(from_logits=True)
        ModelMetrics = ["accuracy"]

        self.trainingModel.compile(loss=lossFct, optimizer="adam", metrics=ModelMetrics)
        
        # create a second model with a softmax layer to get probabilities for AL
        self.softmaxModel = models.Sequential([
            self.trainingModel,
            layers.Softmax()
            ])
    
        # print(trainingModel.summary())



    def trainModel(self, data:ndarray[ndarray[ndarray[uint8]]]=None, labels:ndarray[ndarray[uint8]]=None, verbose:int=None) -> None:
        if data is None or labels is None:
            data = self.xTrainLabeled
            labels = self.yTrainLabeled
            
        if verbose is None:
            verbose = self.verbose
        
        # self.trainingModel.fit(self.xTrainLabeled, self.yTrainLabeled, batch_size=10, epochs=5, shuffle=True, verbose=verbose)
        self.trainingModel.fit(self.xTrainLabeled, self.yTrainLabeled, epochs=10, shuffle=True, verbose=verbose)

        
        
    def createRandomOrder(self, length:int=None) -> list[int]:    # provides an index list for labelData() (random order)
        if length is None:
            length = len(self.yTrainUnlabeled)
    
        order = list(range(length))
        shuffle(order)
        return order
    
    
    
    def singleEntropy(self, array:list[float]) -> float:
        sum = 0
        for value in array:
            if value:
                sum = sum - (value * log2(value))
        return float(sum)

        
        
    def calculateEntropies(self, verbose:int=None) -> list[int]:   # provides an index list for labelData() (entropy sampling)
        if verbose is None:
            verbose = self.verbose
        
        results = self.softmaxModel.predict(self.xTrainUnlabeled, verbose=verbose)  # TODO: (maybe) improve runtime
        
        # TODO: (maybe) improve runtime
        entropies = ndarray(shape=(0,0), dtype=float)
        # Parallel(n_jobs=self.processes, prefer="threads")
        # entropies = append(entropies, Parallel(n_jobs=self.processes)(delayed(self.singleEntropy)(res) for res in results)) # parrallel alternative to the for loop
        for res in results: # calculate entropy for every result
            #res = res/res.sum()    # normalize probabilities
            entropies = append(entropies, self.singleEntropy(res))
            # sum = 0
            # for prob in res:
            #     if prob:
            #         sum = sum - (prob * log2(prob))
            # entropies = append(entropies, sum)
            
        # entropyInds = entropies.argsort()
        return entropies.argsort()[::-1]#entropyInds[::-1]
    
    
    
    # def max_Wrapper(self, i):
    #     return max(self.results[i])
    
    
    
    # TODO: test method
    def calculateLeastConfident(self, verbose:int=None) -> list[int]:    # provides an index list for labelData() (least confident sampling)
        if verbose is None:
                verbose = self.verbose
                
        results = self.softmaxModel.predict(self.xTrainUnlabeled, verbose=verbose)  # TODO: (maybe) improve runtime
        
        maxProb = ndarray(shape=(0,0), dtype=float)
        # maxProb = append(maxProb, Parallel(n_jobs=self.processes)(delayed(self.max_Wrapper)(i) for i in range(len(self.results)))) # parrallel alternative to the for loop (0.95x speed)
        # maxProb = append(maxProb, Parallel(n_jobs=self.processes, prefer="threads")(delayed(max)(res) for res in results)) # parrallel alternative to the for loop (0.85x speed)
        for res in results:
            maxProb = append(maxProb, max(res))
        
        return maxProb.argsort()
    
    
    # TODO: test method
    def calculateMargin(self, verbose:int=None) -> list[int]:    # provides an index list for labelData() (margin sampling)
        if verbose is None:
                verbose = self.verbose
                
        results = self.softmaxModel.predict(self.xTrainUnlabeled, verbose=verbose)
        
        margins = ndarray(shape=(0,0), dtype=float)
        
        for res in results:
            twoLargest = nlargest(2,res)
            margins = append(margins, twoLargest[0]-twoLargest[1])  # abs() not needed, because heapq.nlargest() already sorts its output
        
        return margins.argsort()
    
    
    
    def evaluateModel(self, showIterationNumber:bool=True, verbose:int=None) -> None:
        if verbose is None:
            verbose = self.verbose
        
        if showIterationNumber:
            iterationIndicator = f'of Iteration #{self.completedIterations} '
        else:
            iterationIndicator = ""
        title = f'\n------------------------- Evalutaion {iterationIndicator}-------------------------'
        if verbose > 0:        
            print(title)
        (loss, accuracy) = self.trainingModel.evaluate(self.xTest, self.yTest, batch_size=10, verbose=verbose)
        
        if verbose > 0:
            print("-" * len(title) + "\n")
        
        self.numberLabels.append(len(self.yTrainLabeled))
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        
    
    def labelData(self, sortedIndices:list[int], numDatapoints:int=50) -> None:   # adds new Datapoints from "unlabeled" training data to "labeled" training data
        if numDatapoints > len(self.yTrainUnlabeled):
            print("ERROR: not enough unlabeled data left")
            return
        
        xTrainUnlabeledSorted = self.xTrainUnlabeled[sortedIndices]
        yTrainUnlabeledSorted = self.yTrainUnlabeled[sortedIndices]

        # add most informative Imgs to current Training Data
        self.xTrainLabeled = concatenate((self.xTrainLabeled, xTrainUnlabeledSorted[:numDatapoints]), axis=0)
        self.yTrainLabeled = concatenate((self.yTrainLabeled, yTrainUnlabeledSorted[:numDatapoints]), axis=0)
        self.xTrainUnlabeled = xTrainUnlabeledSorted[numDatapoints:]
        self.yTrainUnlabeled = yTrainUnlabeledSorted[numDatapoints:]



    # def show20img(self, images=None):   # show first 20 images from a given dataset (default: current training data)
    #     if images is None:
    #         images = self.xTrainLabeled
            
    #     plt.figure(figsize=(28, 28))
    #     for i in range(20):
    #         # the number of images in the grid is 4x5 (20)
    #         plt.subplot(4, 5, i+1)
    #         plt.imshow(255-images[i], cmap="gray")
    #         plt.axis("off")
            
            
            
    # def showProgressBar(self, progress, length = 30) -> None:
    #     filled = int(round(length * progress))
    #     bar = "=" * filled + ">" + "." * (length - filled - 1)
        
    #     if filled == length:
    #         stdout.write("\r[" + "=" * filled + "] 100.00%\n")
    #     else:
    #         stdout.write("\r[%s] %.2f%%" % (bar, progress * 100))
            
    #     stdout.flush()
            
            
            
    def activeLearningLoop(self, numIterations:int=1, samplingMethod:str="random", numLabelsPerIteration:int=50, verbose:int=None) -> None:
        if verbose is None:
            verbose = self.verbose
        
        if self.completedIterations < 0:
            self.trainModel(self.xTrainLabeled, self.yTrainLabeled)
            self.completedIterations = 0
            self.evaluateModel()
            
            
        for iterationIndex in range(self.completedIterations, self.completedIterations + numIterations):  # possibly replace with a while loop for different ending criteria
            if samplingMethod.lower() == "random":
                labelingOrder = self.createRandomOrder()
            elif samplingMethod.lower() == "entropy":
                labelingOrder = self.calculateEntropies(verbose=verbose)
            elif samplingMethod.lower() == "leastconfident":
                labelingOrder = self.calculateLeastConfident(verbose=verbose)
            elif samplingMethod.lower() == "margin":
                labelingOrder = self.calculateMargin(verbose=verbose)
            else:
                print("ERROR: unknown sampling method; available options: random, entropy, leastConfident, margin")
                return
            
            self.labelData(labelingOrder, numLabelsPerIteration)
            self.trainModel(self.xTrainLabeled, self.yTrainLabeled, verbose=verbose)
            self.completedIterations = self.completedIterations + 1
            self.evaluateModel(verbose=verbose)
            
            # self.showProgressBar((iterationIndex+1)/numIterations)
            
        # print("Active Learning Loop DONE")

