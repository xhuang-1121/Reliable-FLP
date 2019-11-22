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
iCandidateFaciNum = 30
insName = '30-nodeInstances'

'''
@listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha, 6:boolAllo2Faci]
'''
iGenNum = 100
iPopSize = 50
fCrosRate = 0.9
fMutRate = 0.1
boolAllo2Faci = False
listGAParameters = [iGenNum, iPopSize, iCandidateFaciNum, fCrosRate, fMutRate, fAlpha, boolAllo2Faci]


def funWriteExcel(excelName, a_2d_fEveInsEveRunObjValue):
    rowNum = a_2d_fEveInsEveRunObjValue.shape[0]
    columnNum = a_2d_fEveInsEveRunObjValue.shape[1]
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet('sheet1')
    for i in range(rowNum):
        for j in range(columnNum):
            sheet.write(i, j, a_2d_fEveInsEveRunObjValue[i][j])
    workbook.save(excelName)

def funGA_single(fp_tuple_combOfInsRuns):
    print("Begin: ins ")
    print("Running......")
    cpuStart = time.process_time()
    # 调用GA求解
    GeneticAlgo = GA.GA(listGAParameters, fp_tuple_combOfInsRuns[0])
    finalPop, listGenNum, listfBestIndFitness = GeneticAlgo.funGA_main()
    cpuEnd = time.process_time()
    cpuTime = cpuEnd - cpuStart
    print("End")
    # 记录最终种群中最好个体的fitness和目标函数值，累加
    if listfBestIndFitness[-1] != 1/finalPop[0]['objectValue']:
        print("Wrong. Please check GA.")
    # 为绘图准�?
    new_listfBestIndFitness = [fitness * 1000 for fitness in listfBestIndFitness]
    return cpuTime, listfBestIndFitness[-1], finalPop[0]['objectValue'], new_listfBestIndFitness

def funGA_parallel_8ins():
    '''
    @listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha, 6:boolAllo2Faci]
    '''
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))
    pool = Pool(40)
    list_iRunsIndex = [i for i in range(iRunsNum)]
    plotFile = open('/home/zhanghan/pythonworkspace/reliableFLPm=AllSelcFaci/30-node_GA_poltData(m=#SelectedNodes)-first8.txt', 'a')
    textFile = open('/home/zhanghan/pythonworkspace/reliableFLPm=AllSelcFaci/30-node_GA_EveInsData(m=#SelectedNodes)-first8.txt', 'a')
    f = open(insName, 'rb')
    list_ins = []
    for i in range(iInsNum):
        ins = pickle.load(f)
        list_ins.append(ins)
    listtuple_combOfInsRuns = list(itertools.product(list_ins, list_iRunsIndex))  # int(iInsNum/2) == 4, �?个instances
    # listtuple_combOfInsRuns = list(itertools.product(list_ins[4:], list_iRunsIndex)) # �?个instances
    listtuple_expeResult = pool.map(funGA_single, listtuple_combOfInsRuns)
    pool.close()
    pool.join()
    plt.figure()
    for i in range(iInsNum):
        listfAllDiffGenBestIndFitness = np.zeros((iGenNum + 1,)).tolist()
        for j in range(iRunsNum):
            # 记录CPU time，累�?
            listfAveCPUTimeEveryIns[i] += listtuple_expeResult[i * 10 + j][0]
            # 记录最终种群中最好个体的fitness和目标函数值，累加
            listfAveFitnessEveryIns[i] += listtuple_expeResult[i * 10 + j][1]
            listfAveObjValueEveryIns[i] += listtuple_expeResult[i * 10 + j][2]
            # 为绘图准�?
            new_listfBestIndFitness = listtuple_expeResult[i * 10 + j][3]
            for g in range(len(new_listfBestIndFitness)):
                listfAllDiffGenBestIndFitness[g] += new_listfBestIndFitness[g]
            # 记录每个instance每一次run所得到的最终种群的最优个体的目标函数�?
            a_2d_fEveInsEveRunObjValue[i][j] = listtuple_expeResult[i * 10 + j][2]
        # 平均每次运行的时�?
        print("CPU Time:")
        listfAveCPUTimeEveryIns[i] /= iRunsNum
        print(listfAveCPUTimeEveryIns[i])
        # 平均fitness和目标函数�?
        listfAveFitnessEveryIns[i] /= iRunsNum
        print("Objective Value:")
        listfAveObjValueEveryIns[i] /= iRunsNum
        print(listfAveObjValueEveryIns[i])
        # 绘图
        listfAveBestIndFitnessEveryGen = [fitness / iRunsNum for fitness in listfAllDiffGenBestIndFitness]
        plotFile.write('\nAverage best individual\'s fitness of every generation - ins '+str(i)+':\n')
        plotFile.write(str(listfAveBestIndFitnessEveryGen))
        listGenIndex = list(np.linspace(0, iGenNum, num=(iGenNum + 1)))
        plt.plot(listGenIndex, listfAveBestIndFitnessEveryGen)

    plt.xlabel("# of Generation")
    plt.ylabel("Fitness Of Best Individual (× 1e-3)")
    plt.title("Convergence Curves (10-node, m=#SelectedNodes)")
    plt.savefig("30-node_GA_ConvergenceCurve(m=#SelectedNodes)-first8")
    # 将数据写入text文件
    textFile.write('\nAverage CPU time of 10 runs for each instance:\n')
    textFile.write(str(listfAveCPUTimeEveryIns))
    textFile.write('\n\nAverage fitness of 10 runs for each instance:\n')
    textFile.write(str(listfAveFitnessEveryIns))
    textFile.write('\n\nAverage objective value of 10 runs for each instance:\n')
    textFile.write(str(listfAveObjValueEveryIns))
    np.savetxt("/home/zhanghan/pythonworkspace/reliableFLPm=AllSelcFaci/30-node_GA_ObjValueEveInsEveRun(m=#SelectedNodes)-first8.txt", a_2d_fEveInsEveRunObjValue)
    excelName = '30-node_GA_ObjValueEveInsEveRun(m=#SelectedNodes)-first8.xls'
    funWriteExcel(excelName, a_2d_fEveInsEveRunObjValue)


if __name__ == '__main__':
    funGA_parallel_8ins()
