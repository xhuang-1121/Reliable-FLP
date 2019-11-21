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
fAlpha = 1.0
iCandidateFaciNum = 500
insName = '500-nodeInstances'

def funCplex_cp_single(fp_obInstance):
    listCplexParameters = [iCandidateFaciNum, fAlpha]
    print("Begin: cplex-cp")
    print("Running......")
    cpu_start = time.process_time()
    cplexSolver = usecplex.CPLEX(listCplexParameters, fp_obInstance)
    cplexSolver.fun_fillCpoModel()
    cpsol = cplexSolver.cpomodel.solve()
    cpu_end = time.process_time()
    cpuTime = cpu_end - cpu_start
    fObjectiveValue = cpsol.get_objective_values()[0]
    fGap = cpsol.get_objective_gaps()[0]
    fBound = cpsol.get_objective_bounds()[0]
    print("Solution status: " + cpsol.get_solve_status())
    print("End: cplex-cp")
    return cpuTime, fObjectiveValue, fGap, fBound


def funCplex_cp_parallel():
    afOptimalValueEveIns = np.zeros((iInsNum,))
    listfCpuTimeEveIns = []
    listfGapEveIns = []
    listfBoundEveIns = []

    f = open(insName, 'rb')
    textFile = open('/home/zhanghan/pythonworkspace/reliableFLP/500-node_Cplex_cp_data(m=2).txt', 'a')
    list_ins = []
    for i in range(iInsNum):
        ins = pickle.load(f)
        list_ins.append(ins)
    pool = Pool()
    listtuple_expeResult = pool.map(funCplex_cp_single, list_ins)
    pool.close()
    pool.join()
    if len(listtuple_expeResult) != iInsNum:
        print("Wrong in funCplex_mp_parallel.")

    for i in range(iInsNum):
        listfCpuTimeEveIns.append(listtuple_expeResult[i][0])
        afOptimalValueEveIns[i] = listtuple_expeResult[i][1]
        listfGapEveIns.append(listtuple_expeResult[i][2])
        listfBoundEveIns.append(listtuple_expeResult[i][3])

    textFile.write('\nEvery instance\'s objective value got by Cplex-cp method:\n')
    textFile.write(str(afOptimalValueEveIns))
    textFile.write('\n\nEvery instance\'s CPU time used by Cplex-cp method:\n')
    textFile.write(str(listfCpuTimeEveIns))
    textFile.write('\n\nEvery instance\'s Gap used by Cplex-cp method:\n')
    textFile.write(str(listfGapEveIns))
    textFile.write('\n\nEvery instance\'s Bound used by Cplex-cp method:\n')
    textFile.write(str(listfBoundEveIns))

if __name__ == '__main__':
    # funGA_parallel()
    # funCplex_mp_parallel()
    funCplex_cp_parallel()
    # funCplex_cp_ex()
