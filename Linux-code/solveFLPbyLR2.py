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
iActualInsNum = 8
iInsNum = 8
fAlpha = 1.0
iCandidateFaciNum = 600
insName = '600-nodeInstances'


def funLR2_single(fp_obInstance):
    print("Begin: 600nodes Ins ")
    print("Running......")
    iMaxIterationNum = 600
    fBeta = 2.0
    fBetaMin = 1e-8
    fToleranceEpsilon = 0.001
    boolAllo2Faci = True
    listLRParameters = [iMaxIterationNum, fBeta, fBetaMin, fAlpha, fToleranceEpsilon, boolAllo2Faci]
    cpu_start = time.process_time()
    LagRela = LR2.LagrangianRelaxation(listLRParameters, fp_obInstance)
    LagRela.funInitMultiplierLambda()
    upperBound, lowerBound = LagRela.funLR_main()
    cpu_end = time.process_time()
    cpuTime = cpu_end - cpu_start
    print("UB:", upperBound)
    print("LB:", lowerBound)
    print("End: Ins\n")
    return upperBound, lowerBound, cpuTime


def funLR2_parallel():
    listfUBEveIns = []
    listfLBEveIns = []
    listfCPUTimeEveIns = []
    f = open(insName, 'rb')
    textFile = open('600-node_LR2(m=2).txt', 'a')
    # textFile = open('600-node_LR2(m=all).txt', 'a')
    list_ins = []
    for i in range(iInsNum):
        ins = pickle.load(f)
        list_ins.append(ins)
    pool = Pool()
    listtuple_expeResult = pool.map(funLR2_single, list_ins)
    pool.close()
    pool.join()
    for i in range(iActualInsNum):
        listfUBEveIns.append(listtuple_expeResult[i][0])
        listfLBEveIns.append(listtuple_expeResult[i][1])
        listfCPUTimeEveIns.appen(listtuple_expeResult[i][2])
    textFile.write("\nUpperbound for every instance:\n")
    textFile.write(str(listfUBEveIns))
    textFile.write("\n\nLowerbound for every instance:\n")
    textFile.write(str(listfLBEveIns))
    textFile.write("\n\nCPU time for every instance:\n")
    textFile.write(str(listfCPUTimeEveIns))


def funLR2():
    iMaxIterationNum = 600
    fBeta = 2.0
    fBetaMin = 1e-8
    fToleranceEpsilon = 0.0001
    listLRParameters = [iMaxIterationNum, fBeta, fBetaMin, fAlpha, fToleranceEpsilon]
    listfUBEveIns = []
    listfLBEveIns = []
    f = open(insName, 'rb')
    textFile = open('/home/zhanghan/pythonworkspace/reliableFLPm=2/500-node_LR2(m=2).txt', 'a')
    textFile = open('/home/zhanghan/pythonworkspace/reliableFLPm=AllSelcFaci/500-node_LR2(m=all).txt', 'a')
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
