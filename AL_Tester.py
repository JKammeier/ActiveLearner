# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:59:31 2024

@author: JannisKammeier
"""
import matplotlib.pyplot as plt
from ActiveLearner import ActiveLearner
from time import time
from sys import stdout
from csv import writer
from copy import deepcopy
#import os
#import tensorflow as tf
#from joblib import Parallel, delayed
#from multiprocessing import Pool

runs:int = 20                   # the number of runs of the entire test (to average the results)
initialDatapoints:int = 3000     # the number of datapoints/images that the model will be trained with before starting active learning
labels:int = 200                  # the number of datapoints that will be labeled in each iteration of the active learning cycle
iterations:int = 30         # the number of iterations the active learning cycle will perform

processes = 4

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
    al:ActiveLearner = ActiveLearner(numInitDatapoints=initialDatapoints, processes=processes)
    al.activeLearningLoop(numIterations=iterations, samplingMethod=method, numLabelsPerIteration=labels)
    accuracies:list[float] = deepcopy(al.accuracies)
    del al
    return accuracies


# def loopWrapper(method) -> list[list[float]]:
#     accuracies = []
#     for i in range(iterations):
#         accuracies.append(testWrapper(method))
#     return accuracies




# %% no active learning, initial training on all 60000 datapoints
# al1 = ActiveLearner(numInitDatapoints=-1)


# %% pool based, random sampling
if __name__ == '__main__':
    t1 = time()
    print("Starting random sampling...")
    showProgressBar(0)
    startT = time()
    accuraciesRandom = []
    # accuraciesRandom = Parallel(n_jobs=processes, prefer="threads")(delayed(testWrapper)("random") for i in range(runs))  # parallel loop using joblib
    # with Pool() as pool:
    #     accuraciesRandom = pool.map(testWrapper, ["random" for i in range(runs)])
    for i in range(runs):
        accuraciesRandom.append(testWrapper("random"))
        # al_random = ActiveLearner(numInitDatapoints=initialDatapoints)
        # al_random.activeLearningLoop(numIterations=iterations, samplingMethod="random", numLabelsPerIteration=labels)
        # accuraciesRandom.append(al_random.accuracies)
        showProgressBar((i+1)/runs)
        #print(f"\nAcc: {accuraciesRandom[-1]}")
        
    midT = time()
    avgRandom = averageAccuracies(accuraciesRandom)
    endT = time()
    randRuntime = midT - startT
    randAveraging = endT - midT
    print(f"Random sampling: Runtime = {randRuntime} s; Averaging = {randAveraging} s; total = {randRuntime + randAveraging} s")
    #print(f"Average Accuracy: {avgRandom}")


# %% pool based, least confident sampling
if __name__ == '__main__':
    print("Starting least confident sampling...")
    showProgressBar(0)
    startT = time()
    accuraciesLeastConfident = []
    # accuraciesLeastConfident = Parallel(n_jobs=processes, prefer="threads")(delayed(testWrapper)("leastConfident") for i in range(runs))  # parallel loop using joblib
    # with Pool() as pool:
    #     accuraciesLeastConfident = pool.map(testWrapper, ["leastConfident" for i in range(runs)])
    for i in range(runs):
        accuraciesLeastConfident.append(testWrapper("leastConfident"))
        # al_leastConfident = ActiveLearner(numInitDatapoints=initialDatapoints)
        # al_leastConfident.activeLearningLoop(numIterations=iterations, samplingMethod="leastConfident", numLabelsPerIteration=labels)
        # accuraciesLeastConfident.append(al_leastConfident.accuracies)
        showProgressBar((i+1)/runs)
        
    midT = time()
    avgLeastConfident = averageAccuracies(accuraciesLeastConfident)
    endT = time()
    lcRuntime = midT - startT
    lcAveraging = endT - midT
    print(f"Least confident sampling: Runtime = {lcRuntime} s; Averaging = {lcAveraging} s; total = {lcRuntime + lcAveraging} s")


# %% pool based, margin sampling
if __name__== '__main__':
    print("Starting margin sampling...")
    showProgressBar(0)
    startT = time()
    accuraciesMargin = []
    for i in range(runs):
        accuraciesMargin.append(testWrapper("margin"))
        showProgressBar((i+1)/runs)
    
    midT = time()
    avgMargin = averageAccuracies(accuraciesMargin)
    endT = time()
    margRuntime = midT - startT
    margAveraging = endT - midT
    print(f"Margin sampling: Runtime = {margRuntime} s; Averaging = {margAveraging} s; total = {margRuntime + margAveraging} s")
    

# %% pool based, entropy sampling
if __name__ == '__main__':
    print("Starting entropy sampling...")
    showProgressBar(0)
    startT = time()
    accuraciesEntropy = []
    # accuraciesEntropy = Parallel(n_jobs=processes, prefer="threads")(delayed(testWrapper)("entropy") for i in range(runs))  # parallel loop using joblib
    # with Pool() as pool:
    #     accuraciesEntropy = pool.map(testWrapper, ["entropy" for i in range(runs)])
    for i in range(runs):
        accuraciesEntropy.append(testWrapper("entropy"))
        # al_entropy = ActiveLearner(numInitDatapoints=initialDatapoints)
        # al_entropy.activeLearningLoop(numIterations=iterations, samplingMethod="entropy", numLabelsPerIteration=labels)
        # accuraciesEntropy.append(al_entropy.accuracies)
        showProgressBar((i+1)/runs)
        
    midT = time()
    avgEntropy = averageAccuracies(accuraciesEntropy)
    endT = time()
    entRuntime = midT - startT
    entAveraging = endT - midT
    print(f"Entropy sampling: Runtime = {entRuntime} s; Averaging = {entAveraging} s; total = {entRuntime + entAveraging} s")
    t2 = time()
    print(f"total time: {t2-t1} s")


# %% plot and save results
if __name__ == '__main__':
    timestamp = int(time())
    
    plt.title("comparison of query strategies")
    plt.xlabel("number of labeled datapoints")
    plt.ylabel("accuracy")
    plt.plot(numberLabels, avgEntropy, color="red", label="entropy")
    plt.plot(numberLabels, avgMargin, color="purple", label="margin")
    plt.plot(numberLabels, avgLeastConfident, color="blue", label="least confident")
    plt.plot(numberLabels, avgRandom, color="green", label="random")
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
        
        
    #plt.savefig("ent-lc-rand_5perIt_20runAvg.png")
    #plt.savefig("C:/Users\JannisKammeier/OneDrive - Fachhochschule Bielefeld/Semester_5_Wi22/Studienarbeit/Python/test.png")
    #plt.plot(al2.numberLabels, al2.losses)

# %% shutdown
# if __name__ == '__main__':
    #os.system("shutdown /s /t 30")
