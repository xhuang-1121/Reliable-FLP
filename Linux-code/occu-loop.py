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
iInsNum = 10
iRunsNum = 10
fAlpha = 1.0
iCandidateFaciNum = 500
insName = '500-nodeInstances'

'''
@listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha]
'''
iGenNum = 400
iPopSize = 200
fCrosRate = 0.9
fMutRate = 0.1
boolAllo2Faci = True
listGAParameters = [iGenNum, iPopSize, iCandidateFaciNum, fCrosRate, fMutRate, fAlpha, boolAllo2Faci]


def fun_single(fp_num):
    print("Begin: ins ")
    print("Running......")
    while(True):
        fp_num = fp_num


def fun_parallel_occupation():
    pool = Pool()
    list_iRunsIndex = list(range(iInsNum*iRunsNum))
    f = open(insName, 'rb')
    pool.map(fun_single, list_iRunsIndex)
    pool.close()
    pool.join()
    if len(listtuple_expeResult) != (iInsNum * iRunsNum):
        print("Wrong. Need check.")
    
 
if __name__ == '__main__':
    fun_parallel_occupation()
