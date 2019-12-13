# -*- coding: UTF-8 -*-
import numpy as np
from multiprocessing import Pool
import itertools
import GA
import GA_DM
import GA_LS_DM  # strong local search和weak local search 都在里面
import GAMLS1_DM
import GAMLS2_DM
import GAwithLocalSearch
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
seed = range(30)  # 用于并行程序设置不同的随机种子
iActualInsNum = 8
iInsNum = 8
iRunsNum = 30
fAlpha = 1.0
iCandidateFaciNum = 100
insName = '100-nodeInstances'
fileName = '100-node'

'''
@listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha, 6:boolAllo2Faci]
'''
iGenNum = 50
iPopSize = 50
fCrosRate = 0.9
fMutRate = 0.1
boolAllo2Faci = True
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

def funGAMLS2_DM_single(fp_tuple_combOfInsRuns):
    local_state = np.random.RandomState(fp_tuple_combOfInsRuns[1])
    print("Begin:")
    print("Running......")
    cpuStart = time.process_time()
    # 调用GADM求解
    GeneticAlgo = GAMLS2_DM.GA(listGAParameters, fp_tuple_combOfInsRuns[0], local_state)

    listdictFinalPop, listGenNum, listfBestIndFitnessEveGen, listiDiversityMetric1, listiDiversityMetric2, listiFitEvaNumByThisGen, listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd, listfEveGenProportion_belongToOnlyCurrGenLocalSearchedIndNeighbor, listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd, listiEveGenLocalSearchedIndNum = GeneticAlgo.funGA_main()

    cpuEnd = time.process_time()
    cpuTime = cpuEnd - cpuStart
    print("End")
    # 为绘图准备
    new_listfBestIndFitnessEveGen = [fitness * 1000 for fitness in listfBestIndFitnessEveGen]
    return cpuTime, listfBestIndFitnessEveGen[-1], listdictFinalPop[0]['objectValue'], new_listfBestIndFitnessEveGen, listiDiversityMetric1, listiDiversityMetric2, listiFitEvaNumByThisGen, listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd, listiEveGenLocalSearchedIndNum


def funGAMLS2_DM_parallel():
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))

    textFile = open(fileName + '_GAMLS2_EveInsData_AllRunsAve(m=2).txt', 'a')
    plotFile = open(fileName + '_GAMLS2_PlotData(m=2).txt', 'a')

    list_iRunsIndex = [i for i in range(iRunsNum)]
    f = open(insName, 'rb')
    pool = Pool()
    list_ins = []
    for i in range(iInsNum):
        ins = pickle.load(f)
        list_ins.append(ins)
    listtuple_combOfInsRuns = list(itertools.product(list_ins, list_iRunsIndex))
    listtuple_expeResult = pool.map(funGAMLS2_DM_single, listtuple_combOfInsRuns)  # list中的每个元素都是一个元组，每个元组中存储某instance的某一次run得到的数据
    pool.close()
    pool.join()

    if len(listtuple_expeResult) != (iActualInsNum * iRunsNum):
        print("Wrong. Need check funGAMLS2_DM_parallel().")
    # 整理数据
    for i in range(iActualInsNum):
        listfEveGenBestIndFitness_AllRunsSum = np.zeros((iGenNum + 1,)).tolist()
        listEveGenDiversityMetric1_AllRunsSum = np.zeros((iGenNum + 1,)).tolist()
        listAllRunsAveDiversityMetric2EveGen = np.zeros((iGenNum + 1,)).tolist()
        listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsSum = np.zeros((iGenNum + 1,)).tolist()
        listAllRunsSumFitEvaNumByThisGen = np.zeros((iGenNum + 1,)).tolist()
        listiEveGenLocalSearchedIndNum_AllRunsSum = np.zeros((iGenNum+1,)).tolist()
        for j in range(iRunsNum):
            # 记录CPU time，累加
            listfAveCPUTimeEveryIns[i] += listtuple_expeResult[i * iRunsNum + j][0]
            # 记录最终种群中最好个体的fitness和目标函数值，累加
            listfAveFitnessEveryIns[i] += listtuple_expeResult[i * iRunsNum + j][1]
            listfAveObjValueEveryIns[i] += listtuple_expeResult[i * iRunsNum + j][2]
            # 为绘图准备
            new_listfOneRunBestIndFitnessEveGen = listtuple_expeResult[i * iRunsNum + j][3]
            listiOneRunDiversityMetric1EveGen = listtuple_expeResult[i * iRunsNum + j][4]
            listiOneRunDiversityMetric2EveGen = listtuple_expeResult[i * iRunsNum + j][5]
            listiOneRunFitEvaNumByThisGen = listtuple_expeResult[i*iRunsNum+j][6]
            listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_oneRun = listtuple_expeResult[i*iRunsNum+j][7]
            listiEveGenLocalSearchedIndNum_oneRun = listtuple_expeResult[i*iRunsNum+j][8]

            for g in range(len(new_listfOneRunBestIndFitnessEveGen)):
                listfEveGenBestIndFitness_AllRunsSum[g] += new_listfOneRunBestIndFitnessEveGen[g]
                listEveGenDiversityMetric1_AllRunsSum[g] += listiOneRunDiversityMetric1EveGen[g]
                listAllRunsAveDiversityMetric2EveGen[g] += listiOneRunDiversityMetric2EveGen[g]
                listAllRunsSumFitEvaNumByThisGen[g] += listiOneRunFitEvaNumByThisGen[g]
                listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsSum[g] += listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_oneRun[g]
                listiEveGenLocalSearchedIndNum_AllRunsSum[g] += listiEveGenLocalSearchedIndNum_oneRun[g]
            # 记录每个instance每一次run所得到的最终种群的最优个体的目标函数值
            a_2d_fEveInsEveRunObjValue[i][j] = listtuple_expeResult[i * iRunsNum + j][2]
        # 平均每次运行的时间
        listfAveCPUTimeEveryIns[i] /= iRunsNum
        # 平均fitness和目标函数值
        listfAveFitnessEveryIns[i] /= iRunsNum
        listfAveObjValueEveryIns[i] /= iRunsNum
        # 绘图
        listfEveGenBestIndFitness_AllRunsAve = [fitness / iRunsNum for fitness in listfEveGenBestIndFitness_AllRunsSum]
        listEveGenDiversityMetric1_AllRunsAve = [diversity / iRunsNum for diversity in listEveGenDiversityMetric1_AllRunsSum]
        listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve = [diversity / iRunsNum for diversity in listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsSum]
        listiAveDiversityMetric2EveGen = [diversity / iRunsNum for diversity in listAllRunsAveDiversityMetric2EveGen]
        listiAveFitEvaNumByThisGen = [int(fe / iRunsNum) for fe in listAllRunsSumFitEvaNumByThisGen]
        listiEveGenLocalSearchedIndNum_AllRunsEve = [num / iRunsNum for num in listiEveGenLocalSearchedIndNum_AllRunsSum]

        plotFile.write("listiEveGenLocalSearchedIndNum_AllRunsEve:\n")
        plotFile.write(str(listiEveGenLocalSearchedIndNum_AllRunsEve))
        plotFile.write("\n\nlistfEveGenBestIndFitness_AllRunsAve:\n")
        plotFile.write(str(listfEveGenBestIndFitness_AllRunsAve))
        plotFile.write("\n\nlistEveGenDiversityMetric1_AllRunsAve:\n")
        plotFile.write(str(listEveGenDiversityMetric1_AllRunsAve))
        plotFile.write("\n\nlistfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve:\n")
        plotFile.write(str(listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve))
        plotFile.write("\n\nlistiAveDiversityMetric2EveGen:\n")
        plotFile.write(str(listiAveDiversityMetric2EveGen))
        plotFile.write("\n\nlistiAveFitEvaNumByThisGen:\n")
        plotFile.write(str(listiAveFitEvaNumByThisGen))
        plotFile.write("\n------------------------new instance-----------------------------\n")

        fig = plt.figure()
        listGenIndex = [g for g in range(iGenNum + 1)]
        ax1 = fig.add_subplot(111)
        l1, = ax1.plot(listGenIndex, listfEveGenBestIndFitness_AllRunsAve)
        # 右方Y轴
        ax2 = ax1.twinx()
        l2, = ax2.plot(listGenIndex, listEveGenDiversityMetric1_AllRunsAve, 'r')
        l3, = ax2.plot(listGenIndex, listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve, 'purple', marker='p', linestyle='--')
        # 上方X轴
        ax3 = ax1.twiny()  # 与ax1共用1个y轴，在上方生成自己的x轴
        ax3.set_xlabel("# of Fitness Evaluation")
        listfFeIndex = list(np.linspace(0, iGenNum, num=10+1))
        listFeXCoordinate = []
        for f in range(len(listfFeIndex)):
            listFeXCoordinate.append(listiAveFitEvaNumByThisGen[int(listfFeIndex[f])])
        # print("listFeXCoordinate:", listFeXCoordinate)
        ax3.plot(listGenIndex, listfEveGenBestIndFitness_AllRunsAve, '--')
        ax3.set_xticks(listfFeIndex)
        ax3.set_xticklabels(listFeXCoordinate, rotation=10)
        for label in ax3.xaxis.get_ticklabels():
            label.set_fontsize(8)
        plt.legend(handles=[l1, l2, l3], labels=['l1', 'l2', 'l3'], loc='best')

        ax1.set_xlabel("# of Generation")
        ax1.set_ylabel("Fitness Of Best Individual (× 1e-3)")
        ax2.set_ylabel("Diversity Metric")
        plt.savefig(fileName + '_GAMLS2_Curve(m=2)-ins'+str(i)+'.svg')

    # 将数据写入text文件
    textFile.write('Average CPU time of 10 runs for each instance:\n')
    textFile.write(str(listfAveCPUTimeEveryIns))
    textFile.write('\n\nAverage fitness of 10 runs for each instance:\n')
    textFile.write(str(listfAveFitnessEveryIns))
    textFile.write('\n\nAverage objective value of 10 runs for each instance:\n')
    textFile.write(str(listfAveObjValueEveryIns))
    textFile.write("\n-----------------------------------------------------\n")
    excelName = fileName + '_GAMLS2_ObjValue_EveInsEveRun(m=2).xls'
    funWriteExcel(excelName, a_2d_fEveInsEveRunObjValue)

if __name__ == '__main__':
    # funGA_parallel_4ins()
    # funCplex_mp_parallel()
    # funCplex_cp_parallel()
    # funCplex_cp_ex()
    # funGA_ex()
    # funCplex_mp_ex()
    # funCplex_cp_ex()
    # funGA_DM()
    # funGA_LS_DM()
    # funGA_DM_parallel()
    # funGA_LS_DM_parallel()
    # funGAMLS1_DM_ex()
    # funGAMLS2_DM_ex()
    # funGAMLS1_DM_parallel()
    funGAMLS2_DM_parallel()
