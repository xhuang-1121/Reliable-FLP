# -*- coding: UTF-8 -*-
import numpy as np
from multiprocessing import Pool
import itertools
import GA
import usecplex
import pickle
import instanceGeneration
from instanceGeneration import Instances
import matplotlib.pyplot as plt
import time
import xlwt
import LR1
import LR2

# Global variables
iInsNum = 8
iRunsNum = 10
fAlpha = 1.0
iCandidateFaciNum = 500
insName = '500-nodeInstances'


def funLR2_single(fp_obInstance):
    print("Begin: Ins ")
    print("Running......")
    iMaxIterationNum = 600
    fBeta = 2.0
    fBetaMin = 1e-8
    fToleranceEpsilon = 0.0001
    listLRParameters = [iMaxIterationNum, fBeta, fBetaMin, fAlpha, fToleranceEpsilon]
    LagRela = LR2.LagrangianRelaxation(listLRParameters, fp_obInstance)
    LagRela.funInitMultiplierLambda()
    upperBound, lowerBound = LagRela.funLR_main()
    print("UB:", upperBound)
    print("LB:", lowerBound)
    print("End: Ins\n")
    return upperBound, lowerBound


def funLR2_parallel():
    listfUBEveIns = []
    listfLBEveIns = []
    f = open(insName, 'rb')
    textFile = open('/home/zhanghan/pythonworkspace/reliableFLPm=2/100-node_LR2(m=2).txt', 'a')
    list_ins = []
    for i in range(iInsNum):
        ins = pickle.load(f)
        list_ins.append(ins)
    pool = Pool()
    listtuple_expeResult = pool.map(funLR2_single, list_ins)
    pool.close()
    pool.join()
    for i in range(iInsNum):
        listfUBEveIns.append(listtuple_expeResult[i][0])
        listfLBEveIns.append(listtuple_expeResult[i][1])
    textFile.write("\nUpperbound for every instance:\n")
    textFile.write(str(listfUBEveIns))
    textFile.write("\n\nLowerbound for every instance:\n")
    textFile.write(str(listfLBEveIns))


def funLR2():
    iMaxIterationNum = 600
    fBeta = 2.0
    fBetaMin = 1e-8
    fToleranceEpsilon = 0.0001
    listLRParameters = [iMaxIterationNum, fBeta, fBetaMin, fAlpha, fToleranceEpsilon]
    listfUBEveIns = []
    listfLBEveIns = []
    f = open(insName, 'rb')
    textFile = open('/home/zhanghan/pythonworkspace/reliableFLPm=2/100-node_LR2(m=2).txt', 'a')
    for i in range(iInsNum):
        print("Begin: Ins " + str(i))
        print("Running......")
        ins = pickle.load(f)
        LagRela = LR2.LagrangianRelaxation(listLRParameters, ins)
        LagRela.funInitMultiplierLambda()
        upperBound, lowerBound = LagRela.funLR_main()
        listfUBEveIns.append(upperBound)
        listfLBEveIns.append(lowerBound)
        print("End: Ins " + str(i) + "\n")
    textFile.write("\nUpperbound for every instance:\n")
    textFile.write(str(listfUBEveIns))
    textFile.write("\n\nLowerbound for every instance:\n")
    textFile.write(str(listfLBEveIns))


if __name__ == '__main__':
    funLR2_parallel()
