# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:59:31 2024

@author: JannisKammeier
"""
import matplotlib.pyplot as plt
from ActiveLearner import ActiveLearner
#from ActiveLearner_alternativeTraining import ActiveLearner
from time import time
from sys import stdout
from csv import writer
from copy import deepcopy

t1 = time()

# these variables should be used to set the hyperparameters of the test
runs:int = 10                   # the number of runs of the entire test (to average the results)
initialDatapoints:int = 1000     # the number of datapoints/images that the model will be trained with before starting active learning
iterations:int = 5*4         # the number of iterations the active learning cycle will perform
labels:int = 200                  # the number of datapoints that will be labeled in each iteration of the active learning cycle

# these variables can be used to turn of parts of the testing script
randomSampling:bool = True
leastConfidentSampling:bool = True
marginSampling:bool = True
entropySampling:bool = True
plotAndSaveResults:bool = True


numberLabels:list[int] = list(range(initialDatapoints + (labels*iterations) + 1)[initialDatapoints:(initialDatapoints + (labels*iterations) + 1):labels])



def averageAccuracies(accuracies:list[list[float]]) -> list[float]:
    allAverages = []
    for i in range(len(accuracies[0])):
        avg = 0
        for j in range(len(accuracies)):
            avg = avg + accuracies[j][i]
        avg = avg / len(accuracies)
        allAverages.append(avg)
    return allAverages



def showProgressBar(progress:float, length:int = 50) -> None:
    filled = int(round(length * progress))
    bar = "=" * filled + ">" + "." * (length - filled - 1)
    
    if filled == length:
        stdout.write("\r[" + "=" * filled + "] 100.00%\n")
    else:
        stdout.write("\r[%s] %.2f%%" % (bar, progress * 100))
        
    stdout.flush()
    
    
    
def testWrapper(method:str) -> list[float]:
    al:ActiveLearner = ActiveLearner(numInitDatapoints=initialDatapoints)
    al.activeLearningLoop(numIterations=iterations, samplingMethod=method, numLabelsPerIteration=labels)
    accuracies:list[float] = deepcopy(al.accuracies)
    del al
    return accuracies



# %% no active learning, initial training on all 60000 datapoints
# passiveLearner:ActiveLearner = ActiveLearner(numInitDatapoints=-1)



# %% random sampling
if __name__ == '__main__' and randomSampling:
    
    print("Starting random sampling...")
    showProgressBar(0)
    startT = time()
    accuraciesRandom = []
    for i in range(runs):
        accuraciesRandom.append(testWrapper("random"))
        showProgressBar((i+1)/runs)
        
    avgRandom = averageAccuracies(accuraciesRandom)
    
    randRuntime = time() - startT
    print(f"Random sampling: runtime = {randRuntime} s")



# %% least confident sampling
if __name__ == '__main__' and leastConfidentSampling:
    print("Starting least confident sampling...")
    showProgressBar(0)
    startT = time()
    accuraciesLeastConfident = []
    for i in range(runs):
        accuraciesLeastConfident.append(testWrapper("leastConfident"))
        showProgressBar((i+1)/runs)
        
    avgLeastConfident = averageAccuracies(accuraciesLeastConfident)
    
    lcRuntime = time() - startT
    print(f"Least confident sampling: runtime = {lcRuntime} s")



# %% margin sampling
if __name__== '__main__' and marginSampling:
    print("Starting margin sampling...")
    showProgressBar(0)
    startT = time()
    accuraciesMargin = []
    for i in range(runs):
        accuraciesMargin.append(testWrapper("margin"))
        showProgressBar((i+1)/runs)
    
    avgMargin = averageAccuracies(accuraciesMargin)
    
    margRuntime = time() - startT
    print(f"Margin sampling: runtime = {margRuntime} s")
    
    

# %% entropy sampling
if __name__ == '__main__' and entropySampling:
    print("Starting entropy sampling...")
    showProgressBar(0)
    startT = time()
    accuraciesEntropy = []
    for i in range(runs):
        accuraciesEntropy.append(testWrapper("entropy"))
        showProgressBar((i+1)/runs)
        
    avgEntropy = averageAccuracies(accuraciesEntropy)
    
    entRuntime = time() - startT
    print(f"Entropy sampling: runtime = {entRuntime} s")



# %% plot and save results
if __name__ == '__main__' and plotAndSaveResults:
    timestamp = int(time())
    
    plt.title("comparison of query strategies")
    plt.xlabel("number of labeled datapoints")
    plt.ylabel("accuracy")
    
    if(entropySampling): plt.plot(numberLabels, avgEntropy, color="red", label="entropy")
    if(marginSampling): plt.plot(numberLabels, avgMargin, color="purple", label="margin")
    if(leastConfidentSampling): plt.plot(numberLabels, avgLeastConfident, color="blue", label="least confident")
    if(randomSampling): plt.plot(numberLabels, avgRandom, color="green", label="random")
    
    plt.legend()
    plt.savefig(f"results/{timestamp}_entr-marg-lc-rand_{runs}run_{initialDatapoints}+{labels}-{initialDatapoints + (labels*iterations)}.png")
    
    with open(f"results/{timestamp}_entr-marg-lc-rand_{runs}run_{initialDatapoints}+{labels}-{initialDatapoints + (labels*iterations)}.csv", "w", newline="") as file:
        csv_writer = writer(file)
        csv_writer.writerow(["Number of Labels:"] + numberLabels)
        csv_writer.writerow([])
        csv_writer.writerow(["Entropy"] + avgEntropy)
        csv_writer.writerow([])
        csv_writer.writerow(["Margin"] + avgMargin)
        csv_writer.writerow([])
        csv_writer.writerow(["Least Confident"] + avgLeastConfident)
        csv_writer.writerow([])
        csv_writer.writerow(["Random"] + avgRandom)
        csv_writer.writerow([])
        
        

t2 = time()
print(f"total runtime: {t2-t1} s")

