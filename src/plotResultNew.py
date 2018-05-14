#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import os
import subprocess
from matplotlib.transforms import Bbox

def listToString(p):
    string = ""

    for x in p[0:-1]:
        string = string + str(x) + ","

    string = string + str(p[-1])

    return string

def calcConfInt(p):
    firstElement = p[0]
    allEqual = True

    for element in p:
        if (element != firstElement):
            allEqual = False

    if allEqual:
        return 0

    f = open("tmp.R","w")
    f.write("#!/usr/bin/Rscript\n")

    f.write("print(t.test(c(" + listToString(p) + "),conf.level=0.99))\n")

    f.close()

    os.system("chmod +x ./tmp.R")
    output = subprocess.check_output("./tmp.R", stderr=subprocess.STDOUT, shell=True)

    output = output.split()

    return float(output[31])


team_sizes = [5,10,15,20,25]
n_arms = [100,150,200,250,300]
upper_bounds = [0.30,0.40,0.50,0.60,0.70,0.80,0.90]
results = []
resultsLowCI = []
resultsFolder = "../../results/results_uniform_decay_e"
#expNumber = 0
#bandit_sizes = [100,150,200,250,300]
#us = [0.7,0.8,0.9]

# tau as team size grows
for n in n_arms:
    for u in upper_bounds:
        resultsReward = []
        resultsLowCIReward = []
        resultsPBest = []
        resultsLowCIPBest = []
        resultsTimesBest = []
        resultsLowCITimesBest = []
        resultsCumulativeReward = []
        resultsLowCICumulativeReward = []
        resultsCumulativeRegret = []
        resultsLowCICumulativeRegret = []
        resultsRegretExp = []
        resultsLowCIRegretExp = []
            
            
        for team_sz in team_sizes:
            currentResultReward = []
            currentResultPBest = []
            currentResultTimesBest = []
            currentResultCumulativeReward = []
            currentResultCumulativeRegret = []
            currentResultRegretExp = []

            for expNumber in range(5):            
                pickleFile = open(resultsFolder+"/"+str(expNumber)+"/"+str(n)+"/"+str(team_sz)+"/"+'%.2f' % u+"/results.pickle","rb")
                result = pickle.load(pickleFile)
                pickleFile.close()

                currentResultReward.append(result[12])
                currentResultPBest.append(result[13])
                currentResultTimesBest.append(result[14])
                currentResultCumulativeReward.append(result[15])
                currentResultCumulativeRegret.append(result[16])
                currentResultRegretExp.append(result[17])


            resultsReward.append(np.mean(currentResultReward))
            resultsLowCIReward.append(calcConfInt(currentResultReward))

            resultsPBest.append(np.mean(currentResultPBest))
            resultsLowCIPBest.append(calcConfInt(currentResultPBest))

            resultsTimesBest.append(np.mean(currentResultTimesBest))
            resultsLowCITimesBest.append(calcConfInt(currentResultTimesBest))

            resultsCumulativeReward.append(np.mean(currentResultCumulativeReward))
            resultsLowCICumulativeReward.append(calcConfInt(currentResultCumulativeReward))

            resultsCumulativeRegret.append(np.mean(currentResultCumulativeRegret))
            resultsLowCICumulativeRegret.append(calcConfInt(currentResultCumulativeRegret))

            resultsRegretExp.append(np.mean(currentResultRegretExp))
            resultsLowCIRegretExp.append(calcConfInt(currentResultRegretExp))


        resultsToPlot = [resultsReward,resultsPBest,resultsTimesBest,resultsCumulativeReward,resultsCumulativeRegret,resultsRegretExp]
        CIsToPlot = [resultsLowCIReward,resultsLowCIPBest,resultsLowCITimesBest,resultsLowCICumulativeReward,resultsLowCICumulativeRegret,resultsLowCIRegretExp]
        namesToPlot = ["Reward","PBest","TimesBest","CumulativeReward","CumulativeRegret","CumulativeRegretExp"]
        

        for p in range(6):
            
            plt.figure(figsize=(3.0,2.0))

            plt.errorbar(team_sizes,resultsToPlot[p],yerr=np.array(resultsToPlot[p])-np.array(CIsToPlot[p]))

            plt.xlabel("Team Size")
            plt.ylabel("Tau")

            plt.savefig("plots/tau"+namesToPlot[p]+"-ChangeTeamSize-"+str(n)+"-" + str(u) +"-uniform.pdf",bbox_inches='tight')
            plt.close()

            
# tau as problem size grows

for u in upper_bounds:
    for team_sz in team_sizes:
        resultsReward = []
        resultsLowCIReward = []
        resultsPBest = []
        resultsLowCIPBest = []
        resultsTimesBest = []
        resultsLowCITimesBest = []
        resultsCumulativeReward = []
        resultsLowCICumulativeReward = []
        resultsCumulativeRegret = []
        resultsLowCICumulativeRegret = []
        resultsRegretExp = []
        resultsLowCIRegretExp = []


        for n in n_arms:
            currentResultReward = []
            currentResultPBest = []
            currentResultTimesBest = []
            currentResultCumulativeReward = []
            currentResultCumulativeRegret = []
            currentResultRegretExp = []

            for expNumber in range(5):            
                pickleFile = open(resultsFolder+"/"+str(expNumber)+"/"+str(n)+"/"+str(team_sz)+"/"+'%.2f' % u+"/results.pickle","rb")
                result = pickle.load(pickleFile)
                pickleFile.close()

                currentResultReward.append(result[12])
                currentResultPBest.append(result[13])
                currentResultTimesBest.append(result[14])
                currentResultCumulativeReward.append(result[15])
                currentResultCumulativeRegret.append(result[16])
                currentResultRegretExp.append(result[17])

            resultsReward.append(np.mean(currentResultReward))
            resultsLowCIReward.append(calcConfInt(currentResultReward))

            resultsPBest.append(np.mean(currentResultPBest))
            resultsLowCIPBest.append(calcConfInt(currentResultPBest))

            resultsTimesBest.append(np.mean(currentResultTimesBest))
            resultsLowCITimesBest.append(calcConfInt(currentResultTimesBest))

            resultsCumulativeReward.append(np.mean(currentResultCumulativeReward))
            resultsLowCICumulativeReward.append(calcConfInt(currentResultCumulativeReward))

            resultsCumulativeRegret.append(np.mean(currentResultCumulativeRegret))
            resultsLowCICumulativeRegret.append(calcConfInt(currentResultCumulativeRegret))

            resultsRegretExp.append(np.mean(currentResultRegretExp))
            resultsLowCIRegretExp.append(calcConfInt(currentResultRegretExp))



        resultsToPlot = [resultsReward,resultsPBest,resultsTimesBest,resultsCumulativeReward,resultsCumulativeRegret,resultsRegretExp]
        CIsToPlot = [resultsLowCIReward,resultsLowCIPBest,resultsLowCITimesBest,resultsLowCICumulativeReward,resultsLowCICumulativeRegret,resultsLowCIRegretExp]
        namesToPlot = ["Reward","PBest","TimesBest","CumulativeReward","CumulativeRegret","CumulativeRegretExp"]


        for p in range(6):
            plt.figure(figsize=(3.0,2.0))

            plt.errorbar(n_arms,resultsToPlot[p],yerr=np.array(resultsToPlot[p])-np.array(CIsToPlot[p]))

            plt.xlabel("Problem Size")
            plt.ylabel("Tau")

            plt.savefig("plots/tau"+namesToPlot[p]+"-ChangeProblemSize-"+str(team_sz)+"-" + str(u)+"-uniform.pdf",bbox_inches='tight')
            plt.close()


# # tau as upper bound grows

for n in n_arms:
    for team_sz in team_sizes:
        resultsReward = []
        resultsLowCIReward = []
        resultsPBest = []
        resultsLowCIPBest = []
        resultsTimesBest = []
        resultsLowCITimesBest = []
        resultsCumulativeReward = []
        resultsLowCICumulativeReward = []
        resultsCumulativeRegret = []
        resultsLowCICumulativeRegret = []
        resultsRegretExp = []
        resultsLowCIRegretExp = []


        for u in upper_bounds:
            currentResultReward = []
            currentResultPBest = []
            currentResultTimesBest = []
            currentResultCumulativeReward = []
            currentResultCumulativeRegret = []
            currentResultRegretExp = []


            for expNumber in range(5):            
                pickleFile = open(resultsFolder+"/"+str(expNumber)+"/"+str(n)+"/"+str(team_sz)+"/"+'%.2f' % u+"/results.pickle","rb")
                result = pickle.load(pickleFile)
                pickleFile.close()

                currentResultReward.append(result[12])
                currentResultPBest.append(result[13])
                currentResultTimesBest.append(result[14])
                currentResultCumulativeReward.append(result[15])
                currentResultCumulativeRegret.append(result[16])
                currentResultRegretExp.append(result[17])


            resultsReward.append(np.mean(currentResultReward))
            resultsLowCIReward.append(calcConfInt(currentResultReward))

            resultsPBest.append(np.mean(currentResultPBest))
            resultsLowCIPBest.append(calcConfInt(currentResultPBest))

            resultsTimesBest.append(np.mean(currentResultTimesBest))
            resultsLowCITimesBest.append(calcConfInt(currentResultTimesBest))

            resultsCumulativeReward.append(np.mean(currentResultCumulativeReward))
            resultsLowCICumulativeReward.append(calcConfInt(currentResultCumulativeReward))

            resultsCumulativeRegret.append(np.mean(currentResultCumulativeRegret))
            resultsLowCICumulativeRegret.append(calcConfInt(currentResultCumulativeRegret))

            resultsRegretExp.append(np.mean(currentResultRegretExp))
            resultsLowCIRegretExp.append(calcConfInt(currentResultRegretExp))



        resultsToPlot = [resultsReward,resultsPBest,resultsTimesBest,resultsCumulativeReward,resultsCumulativeRegret,resultsRegretExp]
        CIsToPlot = [resultsLowCIReward,resultsLowCIPBest,resultsLowCITimesBest,resultsLowCICumulativeReward,resultsLowCICumulativeRegret,resultsLowCIRegretExp]
        namesToPlot = ["Reward","PBest","TimesBest","CumulativeReward","CumulativeRegret","CumulativeRegretExp"]

            
        for p in range(6):
            plt.figure(figsize=(3.0,2.0))

            plt.errorbar(upper_bounds,resultsToPlot[p],yerr=np.array(resultsToPlot[p])-np.array(CIsToPlot[p]),capsize=3)

            plt.xlabel("Upper Bound")
            plt.ylabel("Tau")

            plt.savefig("plots/tau"+namesToPlot[p]+"-ChangeUpperBound-"+str(n)+"-" + str(team_sz)+"-uniform.pdf",bbox_inches='tight')
            plt.close()
