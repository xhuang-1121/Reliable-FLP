# -*- coding: UTF-8 -*-
import numpy as np
from multiprocessing import Pool
import itertools
import GA
import GA_DM
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
iActualInsNum = 8
iInsNum = 8
iRunsNum = 10
fAlpha = 1.0
iCandidateFaciNum = 100
insName = '100-nodeInstances'
fileName = '100-node'

'''
@listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha, 6:boolAllo2Faci]
'''
iGenNum = 400
iPopSize = 200
fCrosRate = 0.9
fMutRate = 0.1
boolAllo2Faci = True
listGAParameters = [iGenNum, iPopSize, iCandidateFaciNum, fCrosRate, fMutRate, fAlpha, boolAllo2Faci]


def funWriteExcel(excelName, a_2d_fEveInsEveRunObjValue):
    rowNum = a_2d_fEveInsEveRunObjValue.shape[0]
    columnNum = a_2d_fEveInsEveRunObjValue.shape[1]
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet('sheet1')
    for i, j in itertools.product(range(rowNum), range(columnNum)):
        sheet.write(i, j, a_2d_fEveInsEveRunObjValue[i][j])
    workbook.save(excelName)


def funGA_DM_single(fp_tuple_combOfInsRuns):
    local_state = np.random.RandomState()
    print("Begin:")
    print("Running......")
    cpuStart = time.process_time()
    # 调用GADM求解
    GeneticAlgo = GA_DM.GA(listGAParameters, fp_tuple_combOfInsRuns[0], local_state)
    listdictFinalPop, listGenIndex, listfBestIndFitnessEveGen, listiDiversityMetric1, listiDiversityMetric2, listiFitEvaNumByThisGen = GeneticAlgo.funGA_main()
    cpuEnd = time.process_time()
    cpuTime = cpuEnd - cpuStart
    print("End")
    # 为绘图准备
    new_listfBestIndFitnessEveGen = [fitness * 1000 for fitness in listfBestIndFitnessEveGen]
    return cpuTime, listfBestIndFitnessEveGen[-1], listdictFinalPop[0]['objectValue'], new_listfBestIndFitnessEveGen, listiDiversityMetric1, listiDiversityMetric2, listiFitEvaNumByThisGen


def funGA_DM_parallel():
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))
    textFile = open(f'{fileName}_GADM_EveInsData(m=2).txt', 'a')
    plotFile = open(f'{fileName}_GADM_PlotData(m=2).txt', 'a')
    list_iRunsIndex = list(range(iRunsNum))
    f = open(insName, 'rb')
    pool = Pool()
    list_ins = []
    for _ in range(iInsNum):
        ins = pickle.load(f)
        list_ins.append(ins)
    listtuple_combOfInsRuns = list(itertools.product(list_ins, list_iRunsIndex))
    listtuple_expeResult = pool.map(funGA_DM_single, listtuple_combOfInsRuns)  # list中的每个元素都是一个元组，每个元组中存储某instance的某一次run得到的数据
    pool.close()
    pool.join()
    if len(listtuple_expeResult) != (iActualInsNum * iRunsNum):
        print("Wrong. Need check funGA_DM_parallel().")
    # 整理数据
    for i in range(iActualInsNum):
        listfAllRunsBestIndFitnessEveGen = np.zeros((iGenNum + 1,)).tolist()
        listAllRunsAveDiversityMetric1EveGen = np.zeros((iGenNum + 1,)).tolist()
        listAllRunsAveDiversityMetric2EveGen = np.zeros((iGenNum + 1,)).tolist()
        listAllRunsSumFitEvaNumByThisGen = np.zeros((iGenNum + 1,)).tolist()
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
            for g in range(len(new_listfOneRunBestIndFitnessEveGen)):
                listfAllRunsBestIndFitnessEveGen[g] += new_listfOneRunBestIndFitnessEveGen[g]
                listAllRunsAveDiversityMetric1EveGen[g] += listiOneRunDiversityMetric1EveGen[g]
                listAllRunsAveDiversityMetric2EveGen[g] += listiOneRunDiversityMetric2EveGen[g]
                listAllRunsSumFitEvaNumByThisGen[g] += listiOneRunFitEvaNumByThisGen[g]
            # 记录每个instance每一次run所得到的最终种群的最优个体的目标函数值
            a_2d_fEveInsEveRunObjValue[i][j] = listtuple_expeResult[i * iRunsNum + j][2]
        # 平均每次运行的时间
        listfAveCPUTimeEveryIns[i] /= iRunsNum
        # 平均fitness和目标函数值
        listfAveFitnessEveryIns[i] /= iRunsNum
        listfAveObjValueEveryIns[i] /= iRunsNum
        # 绘图
        listfAveBestIndFitnessEveryGen = [fitness / iRunsNum for fitness in listfAllRunsBestIndFitnessEveGen]
        listiAveDiversityMetric1EveGen = [diversity / iRunsNum for diversity in listAllRunsAveDiversityMetric1EveGen]
        listiAveDiversityMetric2EveGen = [diversity / iRunsNum for diversity in listAllRunsAveDiversityMetric2EveGen]
        listiAveFitEvaNumByThisGen = [int(fe / iRunsNum) for fe in listAllRunsSumFitEvaNumByThisGen]
        plotFile.write("listfAveBestIndFitnessEveryGen:\n")
        plotFile.write(str(listfAveBestIndFitnessEveryGen))
        plotFile.write("\n\nlistiAveDiversityMetric1EveGen:\n")
        plotFile.write(str(listiAveDiversityMetric1EveGen))
        plotFile.write("\n\nlistiAveDiversityMetric2EveGen:\n")
        plotFile.write(str(listiAveDiversityMetric2EveGen))
        plotFile.write("\n\nlistiAveFitEvaNumByThisGen:\n")
        plotFile.write(str(listiAveFitEvaNumByThisGen))
        plotFile.write("\n------------------------new instance-----------------------------\n")

        fig = plt.figure()
        listGenIndex = list(range(iGenNum + 1))
        ax1 = fig.add_subplot(111)
        l1, = ax1.plot(listGenIndex, listfAveBestIndFitnessEveryGen)
        # 右方Y轴
        ax2 = ax1.twinx()
        l2, = ax2.plot(listGenIndex, listiAveDiversityMetric1EveGen, 'r')
        l3, = ax2.plot(listGenIndex, listiAveDiversityMetric2EveGen, 'purple', linestyle='--')
        for label in ax2.yaxis.get_ticklabels():
            label.set_fontsize(6)
        # 上方X轴
        ax3 = ax1.twiny()  # 与ax1共用1个y轴，在上方生成自己的x轴
        ax3.set_xlabel("# of Fitness Evaluation")
        listfFeIndex = list(np.linspace(0, iGenNum, num=10+1))
        listFeXCoordinate = [
            listiAveFitEvaNumByThisGen[int(listfFeIndex[f])]
            for f in range(len(listfFeIndex))
        ]
        # print("listFeXCoordinate:", listFeXCoordinate)
        ax3.plot(listGenIndex, listfAveBestIndFitnessEveryGen)
        ax3.set_xticks(listfFeIndex)
        ax3.set_xticklabels(listFeXCoordinate, rotation=10)
        for label in ax3.xaxis.get_ticklabels():
            label.set_fontsize(6)
        plt.legend(handles=[l1, l2, l3], labels=['Fitness curve', '0-HDR', '1-HDR'], loc='best')

        ax1.set_xlabel("# of Generation")
        ax1.set_ylabel("Fitness Of Best Individual (× 1e-3)")
        ax2.set_ylabel("Diversity Metric")
        plt.savefig(f'{fileName}_GADM_Curve(m=2)-ins{str(i)}.svg')

    '''
    ax1.set_xlabel("# of Generation")
    ax1.set_ylabel("Fitness Of Best Individual (× 1e-3)")
    ax2.set_ylabel("Diversity Metric")
    plt.savefig(fileName + '_GADM_Curve(m=2).svg')
    '''

    # 将数据写入text文件
    textFile.write(
        f'Average CPU time of {str(iRunsNum)}' + ' runs for each instance:\n'
    )
    textFile.write(str(listfAveCPUTimeEveryIns))
    textFile.write('\n\nAverage fitness of '+str(iRunsNum)+' runs for each instance:\n')
    textFile.write(str(listfAveFitnessEveryIns))
    textFile.write('\n\nAverage objective value of '+str(iRunsNum)+' runs for each instance:\n')
    textFile.write(str(listfAveObjValueEveryIns))
    textFile.write("\n-----------------------------------------------------\n")
    excelName = f'{fileName}_GADM_ObjValueEveInsEveRun(m=2).xls'
    funWriteExcel(excelName, a_2d_fEveInsEveRunObjValue)


def funGA_DM():
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))
    textFile = open(f'{fileName}_GADM_EveInsData(m=2).txt', 'a')
    plotFile = open(f'{fileName}_GADM_PlotData(m=2).txt', 'a')
    f = open(insName, 'rb')
    for i in range(iActualInsNum):  # 8 instances
        ins = pickle.load(f)
        # genetic algorithm
        listfAllDiffGenBestIndFitness = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiAllDiffGenDiversityMetric1 = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiAllDiffGenDiversityMetric2 = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        for j in range(iRunsNum):  # Every instance has 10 runs experiments.
            print(f"Begin: ins {str(i)}, Runs {str(i)}")
            print("Running......")
            cpuStart = time.process_time()
            # 调用GADM求解
            GeneticAlgo = GA_DM.GA(listGAParameters, ins)
            listdictFinalPop, listGenNum, listfBestIndFitnessEveGen, listiDiversityMetric1, listiDiversityMetric2 = GeneticAlgo.funGA_main()
            cpuEnd = time.process_time()
            # 记录CPU time，累加
            listfAveCPUTimeEveryIns[i] += (cpuEnd - cpuStart)
            # 记录最终种群中最好个体的fitness和目标函数值，累加
            if listfBestIndFitnessEveGen[-1] != 1/listdictFinalPop[0]['objectValue']:
                print("Wrong. Please check funGA_DM().")
            a_2d_fEveInsEveRunObjValue[i][j] = listdictFinalPop[0]['objectValue']
            listfAveFitnessEveryIns[i] += listfBestIndFitnessEveGen[-1]
            listfAveObjValueEveryIns[i] += listdictFinalPop[0]['objectValue']
            # 为绘图准备
            new_listfBestIndFitness = [fitness * 1000 for fitness in listfBestIndFitnessEveGen]
            for g in range(len(listGenNum)):
                listfAllDiffGenBestIndFitness[g] += new_listfBestIndFitness[g]
                listiAllDiffGenDiversityMetric1[g] += listiDiversityMetric1[g]
                listiAllDiffGenDiversityMetric2[g] += listiDiversityMetric2[g]
        print(f"End: ins {str(i)}, Runs {str(j)}" + "\n")
        # 平均每次运行的时间
        listfAveCPUTimeEveryIns[i] /= iRunsNum
        # 平均fitness和目标函数值
        listfAveFitnessEveryIns[i] /= iRunsNum
        listfAveObjValueEveryIns[i] /= iRunsNum
        # 绘图
        listfAveBestIndFitnessEveryGen = [fitness / iRunsNum for fitness in listfAllDiffGenBestIndFitness]
        listiAveDiversityMetric1EveGen = [diversity / iRunsNum for diversity in listiAllDiffGenDiversityMetric1]
        listiAveDiversityMetric2EveGen = [diversity / iRunsNum for diversity in listiAllDiffGenDiversityMetric2]
        plotFile.write("listfAveBestIndFitnessEveryGen:\n")
        plotFile.write(str(listfAveBestIndFitnessEveryGen))
        plotFile.write("\n\nlistiAveDiversityMetric1EveGen:\n")
        plotFile.write(str(listiAveDiversityMetric1EveGen))
        plotFile.write("\n\nlistiAveDiversityMetric2EveGen:\n")
        plotFile.write(str(listiAveDiversityMetric2EveGen))
        plotFile.write("\n-----------------------------------------------------\n")
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        l1, = ax1.plot(listGenNum, listfAveBestIndFitnessEveryGen)
        ax2 = ax1.twinx()
        l2, = ax2.plot(listGenNum, listiAveDiversityMetric1EveGen, 'r')
        l3, = ax2.plot(listGenNum, listiAveDiversityMetric2EveGen, 'purple', linestyle='--')
        plt.legend(handles=[l1, l2, l3], labels=['Fitness curve', '0-HDR', '1-HDR'], loc='best')

    # plt.xlabel("# of Generation")
    ax1.set_xlabel("# of Generation")
    ax1.set_ylabel("Fitness Of Best Individual (× 1e-3)")
    ax2.set_ylabel("Diversity Metric")
    plt.savefig(f'{fileName}_GADM_Curve(m=2).svg')
    # 将数据写入text文件
    textFile.write(
        f'Average CPU time of {str(iRunsNum)}' + ' runs for each instance:\n'
    )
    textFile.write(str(listfAveCPUTimeEveryIns))
    textFile.write('\n\nAverage fitness of '+str(iRunsNum)+' runs for each instance:\n')
    textFile.write(str(listfAveFitnessEveryIns))
    textFile.write('\n\nAverage objective value of '+str(iRunsNum)+' runs for each instance:\n')
    textFile.write(str(listfAveObjValueEveryIns))
    textFile.write("\n-----------------------------------------------------\n")
    # np.savetxt("100-node_GA_ObjValueEveInsEveRun(m=2).txt", a_2d_fEveInsEveRunObjValue)
    excelName = f'{fileName}_GADM_ObjValueEveInsEveRun(m=2).xls'
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
    funGA_DM_parallel()
