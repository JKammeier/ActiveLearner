# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:59:31 2024

@author: JannisKammeier
"""
import matplotlib.pyplot as plt
from ActiveLearner import activeLearner
from time import time
from sys import stdout
#import os
#import tensorflow as tf
#from joblib import Parallel, delayed
#from multiprocessing import Pool

runs = 10
initialDatapoints = 100
labels = 5
iterations = 20 * 4

processes = 4

numberLabels = range(initialDatapoints + (labels*iterations) + 1)[initialDatapoints:(initialDatapoints + (labels*iterations) + 1):labels]



def averageAccuracies(accuracies):
    allAverages = []
    for i in range(len(accuracies[0])):
        avg = 0
        for j in range(len(accuracies)):
            avg = avg + accuracies[j][i]
        avg = avg / len(accuracies)
        allAverages.append(avg)
    return allAverages


def showProgressBar(progress, length = 50):
    filled = int(round(length * progress))
    bar = "=" * filled + ">" + "." * (length - filled - 1)
    
    if filled == length:
        stdout.write("\r[" + "=" * filled + "] 100.00%\n")
    else:
        stdout.write("\r[%s] %.2f%%" % (bar, progress * 100))
        
    stdout.flush()
    
def testWrapper(method):
    al = activeLearner(numInitDatapoints=initialDatapoints, processes=processes)
    al.activeLearningLoop(numIterations=iterations, samplingMethod=method, numLabelsPerIteration=labels)
    return al.accuracies


def loopWrapper(method):
    accuracies = []
    for i in range(iterations):
        accuracies.append(testWrapper(method))
    return accuracies




# %% no active learning, initial training on all 60000 datapoints
# al1 = activeLearner(numInitDatapoints=-1)



# %% pool based, random sampling, datapoints: 50 inital; 10 per iteration
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
        # al_random = activeLearner(numInitDatapoints=initialDatapoints)
        # al_random.activeLearningLoop(numIterations=iterations, samplingMethod="random", numLabelsPerIteration=labels)
        # accuraciesRandom.append(al_random.accuracies)
        showProgressBar((i+1)/runs)
        
    midT = time()
    avgRandom = averageAccuracies(accuraciesRandom)
    endT = time()
    randRuntime = midT - startT
    randAveraging = endT - midT
    print(f"Random sampling: Runtime = {randRuntime} s; Averaging = {randAveraging} s; total = {randRuntime + randAveraging} s")


# %% pool based, least confident sampling, datapoints: 50 initial; 10 per iteration
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
        # al_leastConfident = activeLearner(numInitDatapoints=initialDatapoints)
        # al_leastConfident.activeLearningLoop(numIterations=iterations, samplingMethod="leastConfident", numLabelsPerIteration=labels)
        # accuraciesLeastConfident.append(al_leastConfident.accuracies)
        showProgressBar((i+1)/runs)
        
    midT = time()
    avgLeastConfident = averageAccuracies(accuraciesLeastConfident)
    endT = time()
    lcRuntime = midT - startT
    lcAveraging = endT - midT
    print(f"Least Confident sampling: Runtime = {lcRuntime} s; Averaging = {lcAveraging} s; total = {lcRuntime + lcAveraging} s")


# %% pool based, entropy sampling, datapoints: 50 initial; 10 per iteration
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
        # al_entropy = activeLearner(numInitDatapoints=initialDatapoints)
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


# %% plot
if __name__ == '__main__':
    plt.title("comparison of query strategies")
    plt.xlabel("number of labeled datapoints")
    plt.ylabel("accuracy")
    plt.plot(numberLabels, avgEntropy, color="red", label="entropy")
    plt.plot(numberLabels, avgLeastConfident, color="green", label="least confident")
    plt.plot(numberLabels, avgRandom, color="blue", label="random")
    plt.legend()
    plt.savefig("entr-lc-rand_10run_100+5-500.png")
    #plt.savefig("ent-lc-rand_5perIt_20runAvg.png")
    #plt.savefig("C:/Users\JannisKammeier/OneDrive - Fachhochschule Bielefeld/Semester_5_Wi22/Studienarbeit/Python/test.png")
    #plt.plot(al2.numberLabels, al2.losses)

# %% shutdown
# if __name__ == '__main__':
    #os.system("shutdown /s /t 30")