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
fAlpha = 1.0
iCandidateFaciNum = 100
insName = '100-nodeInstances'
boolAllo2Faci = False

def funCplex_mp_single(fp_obInstance):
    listCplexParameters = [iCandidateFaciNum, fAlpha]
    textFile = open('/home/zhanghan/pythonworkspace/reliableFLPm=AllSelcFaci/100-node_Cplex_mp_data(m=all).txt', 'a')
    print("Begin: Ins ")
    print("Running......")
    cpu_start = time.process_time()
    cplexSolver = usecplex.CPLEX(listCplexParameters, fp_obInstance)
    cplexSolver.fun_fillMpModel_AlloAllSelcFaci()
    sol = cplexSolver.model.solve()
    cpu_end = time.process_time()
    cpuTime = cpu_end - cpu_start
    fOptimalValue = sol.get_objective_value()
    print(sol.solve_details)
    print("Optimal value:", fOptimalValue)
    textFile.write('\nEvery instance\'s objective value got by Cplex-mp method:\n')
    textFile.write(str(fOptimalValue))
    textFile.write('\n\nEvery instance\'s CPU time used by Cplex-mp method:\n')
    textFile.write(str(cpuTime))
    print("End: Ins ")
    return cpuTime, fOptimalValue


def funCplex_mp_parallel():
    afOptimalValueEveIns = np.zeros((iInsNum,))
    listfCpuTimeEveIns = []

    f = open(insName, 'rb')
    textFile = open('/home/zhanghan/pythonworkspace/reliableFLPm=AllSelcFaci/100-node_Cplex_mp_data(m=all).txt', 'a')
    list_ins = []
    for _ in range(iInsNum):
        ins = pickle.load(f)
        list_ins.append(ins)
    pool = Pool()
    listtuple_expeResult = pool.map(funCplex_mp_single, list_ins)
    pool.close()
    pool.join()
    if len(listtuple_expeResult) != iInsNum:
        print("Wrong in funCplex_mp_parallel.")

    for i in range(iInsNum):
        listfCpuTimeEveIns.append(listtuple_expeResult[i][0])
        afOptimalValueEveIns[i] = listtuple_expeResult[i][1]

    textFile.write('\nEvery instance\'s objective value got by Cplex-mp method:\n')
    textFile.write(str(afOptimalValueEveIns))
    textFile.write('\n\nEvery instance\'s CPU time used by Cplex-mp method:\n')
    textFile.write(str(listfCpuTimeEveIns))



# funGA()
# funCplex_mp()
# funLR2()
# funLR1()
# funGA_ex()
# funCplex_cp_ex()
if __name__ == '__main__':
    funCplex_mp_parallel()
