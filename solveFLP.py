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
seed = range(10)  # 用于并行程序设置不同的随机种子
iActualInsNum = 8
iInsNum = 8
iRunsNum = 8
fAlpha = 1.0
iCandidateFaciNum = 30
insName = '30-nodeInstances'
fileName = 'ex30-node'

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


def funGA_DM_single(fp_tuple_combOfInsRuns):
    print("Begin:")
    print("Running......")
    cpuStart = time.process_time()
    # 调用GADM求解
    GeneticAlgo = GA_DM.GA(listGAParameters, fp_tuple_combOfInsRuns[0])
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
    textFile = open(fileName + '_GADM_EveInsData(m=2).txt', 'a')
    plotFile = open(fileName + '_GADM_PlotData(m=2).txt', 'a')
    list_iRunsIndex = [i for i in range(iRunsNum)]
    f = open(insName, 'rb')
    pool = Pool()
    list_ins = []
    for i in range(iInsNum):
        ins = pickle.load(f)
        list_ins.append(ins)
    listtuple_combOfInsRuns = list(itertools.product(list_ins[0:1], list_iRunsIndex))
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
        listGenIndex = [i for i in range(iGenNum + 1)]
        ax1 = fig.add_subplot(111)
        l1, = ax1.plot(listGenIndex, listfAveBestIndFitnessEveryGen)
        # 右方Y轴
        ax2 = ax1.twinx()
        l2, = ax2.plot(listGenIndex, listiAveDiversityMetric1EveGen, 'r')
        l3, = ax2.plot(listGenIndex, listiAveDiversityMetric2EveGen, 'purple', linestyle='--')
        # 上方X轴
        ax3 = ax1.twiny()  # 与ax1共用1个y轴，在上方生成自己的x轴
        ax3.set_xlabel("# of Fitness Evaluation")
        listfFeIndex = list(np.linspace(0, iGenNum, num=10+1))
        # print("listFeIndex:", listfFeIndex)
        listFeXCoordinate = []
        for i in range(len(listfFeIndex)):
            listFeXCoordinate.append(listiAveFitEvaNumByThisGen[int(listfFeIndex[i])])
        # print("listFeXCoordinate:", listFeXCoordinate)
        ax3.plot(listGenIndex, listfAveBestIndFitnessEveryGen)
        ax3.set_xticks(listfFeIndex)
        ax3.set_xticklabels(listFeXCoordinate, rotation=10)
        for label in ax3.xaxis.get_ticklabels():
            label.set_fontsize(6)
        plt.legend(handles=[l1, l2, l3], labels=['Fitness curve', 'Diversity curve - No neighborhood', 'Diversity curve - With neighborhood'], loc='best')

    # plt.xlabel("# of Generation")
    ax1.set_xlabel("# of Generation")
    ax1.set_ylabel("Fitness Of Best Individual (× 1e-3)")
    ax2.set_ylabel("Diversity Metric")
    plt.savefig(fileName + '_GADM_Curve(m=2).svg')
    # 将数据写入text文件
    textFile.write('Average CPU time of 10 runs for each instance:\n')
    textFile.write(str(listfAveCPUTimeEveryIns))
    textFile.write('\n\nAverage fitness of 10 runs for each instance:\n')
    textFile.write(str(listfAveFitnessEveryIns))
    textFile.write('\n\nAverage objective value of 10 runs for each instance:\n')
    textFile.write(str(listfAveObjValueEveryIns))
    textFile.write("\n-----------------------------------------------------\n")
    excelName = fileName + '_GADM_ObjValueEveInsEveRun(m=2).xls'
    funWriteExcel(excelName, a_2d_fEveInsEveRunObjValue)


def funGA_DM():
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))
    textFile = open(fileName + '_GADM_EveInsData(m=2).txt', 'a')
    plotFile = open(fileName + '_GADM_PlotData(m=2).txt', 'a')
    f = open(insName, 'rb')
    for i in range(iActualInsNum):  # 8 instances
        ins = pickle.load(f)
        # genetic algorithm
        listfAllDiffGenBestIndFitness = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiAllDiffGenDiversityMetric1 = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiAllDiffGenDiversityMetric2 = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        for j in range(iRunsNum):  # Every instance has 10 runs experiments.
            print("Begin: ins " + str(i) + ", Runs " + str(i))
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
        print("End: ins " + str(i) + ", Runs " + str(j) + "\n")
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
        plt.legend(handles=[l1, l2, l3], labels=['Fitness curve', 'Diversity curve - No neighborhood', 'Diversity curve - With neighborhood'], loc='best')

    # plt.xlabel("# of Generation")
    ax1.set_xlabel("# of Generation")
    ax1.set_ylabel("Fitness Of Best Individual (× 1e-3)")
    ax2.set_ylabel("Diversity Metric")
    plt.savefig(fileName + '_GADM_Curve(m=2).svg')
    # 将数据写入text文件
    textFile.write('Average CPU time of 10 runs for each instance:\n')
    textFile.write(str(listfAveCPUTimeEveryIns))
    textFile.write('\n\nAverage fitness of 10 runs for each instance:\n')
    textFile.write(str(listfAveFitnessEveryIns))
    textFile.write('\n\nAverage objective value of 10 runs for each instance:\n')
    textFile.write(str(listfAveObjValueEveryIns))
    textFile.write("\n-----------------------------------------------------\n")
    # np.savetxt("100-node_GA_ObjValueEveInsEveRun(m=2).txt", a_2d_fEveInsEveRunObjValue)
    excelName = fileName + '_GADM_ObjValueEveInsEveRun(m=2).xls'
    funWriteExcel(excelName, a_2d_fEveInsEveRunObjValue)


def funGA_LS_DM_single(fp_tuple_combOfInsRuns):
    print("Begin:")
    print("Running......")
    cpuStart = time.process_time()
    # 调用GADM求解
    GeneticAlgo = GA_LS_DM.GA(listGAParameters, fp_tuple_combOfInsRuns[0])
    listdictFinalPop, listGenIndex, listfBestIndFitnessEveGen, listiDiversityMetric1, listiDiversityMetric2, listiFitEvaNumByThisGen = GeneticAlgo.funGA_main()
    cpuEnd = time.process_time()
    cpuTime = cpuEnd - cpuStart
    print("End")
    # 为绘图准备
    new_listfBestIndFitnessEveGen = [fitness * 1000 for fitness in listfBestIndFitnessEveGen]
    return cpuTime, listfBestIndFitnessEveGen[-1], listdictFinalPop[0]['objectValue'], new_listfBestIndFitnessEveGen, listiDiversityMetric1, listiDiversityMetric2, listiFitEvaNumByThisGen


def funGA_LS_DM_parallel():
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))
    textFile = open(fileName + '_GALSDM_EveInsData(m=2).txt', 'a')
    plotFile = open(fileName + '_GALSDM_PlotData(m=2).txt', 'a')
    list_iRunsIndex = [i for i in range(iRunsNum)]
    f = open(insName, 'rb')
    pool = Pool()
    list_ins = []
    for i in range(iInsNum):
        ins = pickle.load(f)
        list_ins.append(ins)
    listtuple_combOfInsRuns = list(itertools.product(list_ins[0:1], list_iRunsIndex))
    listtuple_expeResult = pool.map(funGA_LS_DM_single, listtuple_combOfInsRuns)  # list中的每个元素都是一个元组，每个元组中存储某instance的某一次run得到的数据
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
        listGenIndex = [i for i in range(iGenNum + 1)]
        ax1 = fig.add_subplot(111)
        l1, = ax1.plot(listGenIndex, listfAveBestIndFitnessEveryGen)
        # 右方Y轴
        ax2 = ax1.twinx()
        l2, = ax2.plot(listGenIndex, listiAveDiversityMetric1EveGen, 'r')
        l3, = ax2.plot(listGenIndex, listiAveDiversityMetric2EveGen, 'purple', linestyle='--')
        # 上方X轴
        ax3 = ax1.twiny()  # 与ax1共用1个y轴，在上方生成自己的x轴
        ax3.set_xlabel("# of Fitness Evaluation")
        listfFeIndex = list(np.linspace(0, iGenNum, num=10+1))
        # print("listFeIndex:", listfFeIndex)
        listFeXCoordinate = []
        for i in range(len(listfFeIndex)):
            listFeXCoordinate.append(listiAveFitEvaNumByThisGen[int(listfFeIndex[i])])
        # print("listFeXCoordinate:", listFeXCoordinate)
        ax3.plot(listGenIndex, listfAveBestIndFitnessEveryGen)
        ax3.set_xticks(listfFeIndex)
        ax3.set_xticklabels(listFeXCoordinate, rotation=10)
        for label in ax3.xaxis.get_ticklabels():
            label.set_fontsize(6)
        plt.legend(handles=[l1, l2, l3], labels=['Fitness curve', 'Diversity curve - No neighborhood', 'Diversity curve - With neighborhood'], loc='best')

    # plt.xlabel("# of Generation")
    ax1.set_xlabel("# of Generation")
    ax1.set_ylabel("Fitness Of Best Individual (× 1e-3)")
    ax2.set_ylabel("Diversity Metric")
    plt.savefig(fileName + '_GALSDM_Curve(m=2).svg')
    # 将数据写入text文件
    textFile.write('Average CPU time of 10 runs for each instance:\n')
    textFile.write(str(listfAveCPUTimeEveryIns))
    textFile.write('\n\nAverage fitness of 10 runs for each instance:\n')
    textFile.write(str(listfAveFitnessEveryIns))
    textFile.write('\n\nAverage objective value of 10 runs for each instance:\n')
    textFile.write(str(listfAveObjValueEveryIns))
    textFile.write("\n-----------------------------------------------------\n")
    excelName = fileName + '_GALSDM_ObjValueEveInsEveRun(m=2).xls'
    funWriteExcel(excelName, a_2d_fEveInsEveRunObjValue)


def funGA_LS_DM():
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))
    textFile = open(fileName + '_GALSDM_EveInsData(m=2).txt', 'a')
    plotFile = open(fileName + '_GALSDM_PlotData(m=2).txt', 'a')
    f = open(insName, 'rb')
    fig = plt.figure()
    for i in range(iActualInsNum):  # 8 instances
        ins = pickle.load(f)
        # genetic algorithm
        listfAllDiffGenBestIndFitness = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiAllDiffGenDiversityMetric1 = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiAllDiffGenDiversityMetric2 = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        for j in range(iRunsNum):  # Every instance has 10 runs experiments.
            print("Begin: ins " + str(i) + ", Runs " + str(i))
            print("Running......")
            cpuStart = time.process_time()
            # 调用GADM求解
            GeneticAlgo = GA_LS_DM.GA(listGAParameters, ins)
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
        print("End: ins " + str(i) + ", Runs " + str(j) + "\n")
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

        ax1 = fig.add_subplot(111)
        l1, = ax1.plot(listGenNum, listfAveBestIndFitnessEveryGen)
        ax2 = ax1.twinx()
        l2, = ax2.plot(listGenNum, listiAveDiversityMetric1EveGen, 'r')
        l3, = ax2.plot(listGenNum, listiAveDiversityMetric2EveGen, 'purple', linestyle='--')
        plt.legend(handles=[l1, l2, l3], labels=['Fitness curve', 'Diversity curve - No neighborhood', 'Diversity curve - With neighborhood'], loc='best')

    # plt.xlabel("# of Generation")
    ax1.set_xlabel("# of Generation")
    ax1.set_ylabel("Fitness Of Best Individual (× 1e-3)")
    ax2.set_ylabel("Diversity Metric")
    plt.savefig(fileName + '_GALSDM_Curve(m=2).svg')
    # 将数据写入text文件
    textFile.write('Average CPU time of 10 runs for each instance:\n')
    textFile.write(str(listfAveCPUTimeEveryIns))
    textFile.write('\n\nAverage fitness of 10 runs for each instance:\n')
    textFile.write(str(listfAveFitnessEveryIns))
    textFile.write('\n\nAverage objective value of 10 runs for each instance:\n')
    textFile.write(str(listfAveObjValueEveryIns))
    textFile.write("\n-----------------------------------------------------\n")
    # np.savetxt("100-node_GA_ObjValueEveInsEveRun(m=2).txt", a_2d_fEveInsEveRunObjValue)
    excelName = fileName + '_GALSDM_ObjValueEveInsEveRun(m=2).xls'
    funWriteExcel(excelName, a_2d_fEveInsEveRunObjValue)


def funGAMLS1_DM_single(fp_tuple_combOfInsRuns):
    local_state = np.random.RandomState(fp_tuple_combOfInsRuns[1])
    print("Begin:")
    print("Running......")
    cpuStart = time.process_time()
    # 调用GADM求解
    GeneticAlgo = GAMLS1_DM.GA(listGAParameters, fp_tuple_combOfInsRuns[0], local_state)

    listdictFinalPop, listGenNum, listfBestIndFitnessEveGen, listiDiversityMetric1, listiDiversityMetric2, listiFitEvaNumByThisGen, listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd, listfEveGenProportion_belongToOnlyCurrGenLocalSearchedIndNeighbor, listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd, listiEveGenLocalSearchedIndNum = GeneticAlgo.funGA_main()

    cpuEnd = time.process_time()
    cpuTime = cpuEnd - cpuStart
    print("End")
    # 为绘图准备
    new_listfBestIndFitnessEveGen = [fitness * 1000 for fitness in listfBestIndFitnessEveGen]
    return cpuTime, listfBestIndFitnessEveGen[-1], listdictFinalPop[0]['objectValue'], new_listfBestIndFitnessEveGen, listiDiversityMetric1, listiDiversityMetric2, listiFitEvaNumByThisGen, listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd, listiEveGenLocalSearchedIndNum


def funGAMLS1_DM_parallel():
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))

    textFile = open(fileName + '_GAMLS1_EveInsData_AllRunsAve(m=2).txt', 'a')
    plotFile = open(fileName + '_GAMLS1_PlotData(m=2).txt', 'a')

    list_iRunsIndex = [i for i in range(iRunsNum)]
    f = open(insName, 'rb')
    pool = Pool()
    list_ins = []
    for i in range(iInsNum):
        ins = pickle.load(f)
        list_ins.append(ins)
    listtuple_combOfInsRuns = list(itertools.product(list_ins[0:1], list_iRunsIndex))
    listtuple_expeResult = pool.map(funGAMLS1_DM_single, listtuple_combOfInsRuns)  # list中的每个元素都是一个元组，每个元组中存储某instance的某一次run得到的数据
    pool.close()
    pool.join()

    if len(listtuple_expeResult) != (iActualInsNum * iRunsNum):
        print("Wrong. Need check funGAMLS1_DM_parallel().")
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
        plt.savefig(fileName + '_GAMLS1_Curve(m=2)-ins'+str(i)+'.svg')

    # 将数据写入text文件
    textFile.write('Average CPU time of 10 runs for each instance:\n')
    textFile.write(str(listfAveCPUTimeEveryIns))
    textFile.write('\n\nAverage fitness of 10 runs for each instance:\n')
    textFile.write(str(listfAveFitnessEveryIns))
    textFile.write('\n\nAverage objective value of 10 runs for each instance:\n')
    textFile.write(str(listfAveObjValueEveryIns))
    textFile.write("\n-----------------------------------------------------\n")
    excelName = fileName + '_GAMLS1_ObjValue_EveInsEveRun(m=2).xls'
    funWriteExcel(excelName, a_2d_fEveInsEveRunObjValue)


def funGAMLS1_DM_ex():
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    # listAllRunsSumFitEvaNumByThisGen = np.zeros((iGenNum + 1,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))
    f = open(insName, 'rb')
    listIns = []
    fig = plt.figure()
    for i in range(iInsNum):  # 8 instances
        ins = pickle.load(f)
        listIns.append(ins)
    for i in range(iActualInsNum):
        # genetic algorithm
        listfAllDiffGenBestIndFitness = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiAllDiffGenDiversityMetric1 = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiAllDiffGenDiversityMetric2 = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiFitEvaNumByThisGen_AllRunsAve = np.zeros((iGenNum + 1,)).tolist()
        listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve = np.zeros((iGenNum + 1,)).tolist()
        listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve = np.zeros((iGenNum + 1,)).tolist()
        for j in range(iRunsNum):  # Every instance has 10 runs experiments.
            print("Begin: ins " + str(i) + ", Runs " + str(j))
            print("Running......")
            cpuStart = time.process_time()
            # 调用GADM求解
            local_state = np.random.RandomState()
            GeneticAlgo = GAMLS1_DM.GA(listGAParameters, listIns[3], local_state)
            listdictFinalPop, listGenNum, listfBestIndFitnessEveGen, listiDiversityMetric1, listiDiversityMetric2, listiFitEvaNumByThisGen, listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd, listfEveGenProportion_belongToOnlyCurrGenLocalSearchedIndNeighbor, listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd = GeneticAlgo.funGA_main()
            cpuEnd = time.process_time()
            # 记录CPU time，累加
            listfAveCPUTimeEveryIns[i] += (cpuEnd - cpuStart)
            # 记录最终种群中最好个体的fitness和目标函数值，累加
            print("Objective value:", listdictFinalPop[0]['objectValue'])
            if listfBestIndFitnessEveGen[-1] != 1/listdictFinalPop[0]['objectValue']:
                print("Wrong. Please check funGA_LS_DM().")
            a_2d_fEveInsEveRunObjValue[i][j] = listdictFinalPop[0]['objectValue']
            listfAveFitnessEveryIns[i] += listfBestIndFitnessEveGen[-1]
            listfAveObjValueEveryIns[i] += listdictFinalPop[0]['objectValue']
            # 为绘图准备
            new_listfBestIndFitness = [fitness * 1000 for fitness in listfBestIndFitnessEveGen]
            for g in range(len(listGenNum)):
                listfAllDiffGenBestIndFitness[g] += new_listfBestIndFitness[g]
                listiAllDiffGenDiversityMetric1[g] += listiDiversityMetric1[g]
                listiAllDiffGenDiversityMetric2[g] += listiDiversityMetric2[g]
                listiFitEvaNumByThisGen_AllRunsAve[g] += listiFitEvaNumByThisGen[g]
                listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve[g] += listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd[g]
                listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve[g] += listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd[g]

        print("End: ins " + str(i) + ", Runs " + str(j) + "\n")
        # 平均每次运行的时间
        listfAveCPUTimeEveryIns[i] /= iRunsNum
        # 平均fitness和目标函数值
        listfAveFitnessEveryIns[i] /= iRunsNum
        listfAveObjValueEveryIns[i] /= iRunsNum
        # 绘图
        listfAveBestIndFitnessEveryGen = [fitness / iRunsNum for fitness in listfAllDiffGenBestIndFitness]
        listiAveDiversityMetric1EveGen = [diversity / iRunsNum for diversity in listiAllDiffGenDiversityMetric1]
        listiAveDiversityMetric2EveGen = [diversity / iRunsNum for diversity in listiAllDiffGenDiversityMetric2]
        listiFitEvaNumByThisGen_AllRunsAve = [diversity / iRunsNum for diversity in listiFitEvaNumByThisGen_AllRunsAve]
        listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve = [diversity / iRunsNum for diversity in listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve]
        listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve = [diversity / iRunsNum for diversity in listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve]

        ax1 = fig.add_subplot(111)
        l1, = ax1.plot(listGenNum, listfAveBestIndFitnessEveryGen, marker='*')
        ax1.set_xlabel("# of Generation")
        ax1.set_ylabel("Fitness Of Best Individual")
        # 右方Y轴
        ax2 = ax1.twinx()  # 与ax1共用1个x轴，在右方生成自己的y轴
        l2, = ax2.plot(listGenNum, listiAveDiversityMetric1EveGen, 'r')
        l3, = ax2.plot(listGenNum, listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve, 'purple', marker='p', linestyle='--')
        # l4, = ax2.plot(listGenNum, listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve, 'darkslategray', linestyle='-.')
        ax2.set_ylabel("Diversity Metric")
        # 上方X轴
        ax3 = ax1.twiny()  # 与ax1共用1个y轴，在上方生成自己的x轴
        ax3.set_xlabel("# of Fitness Evaluation")
        listfFeIndex = list(np.linspace(0, iGenNum, num=10+1))
        listFeXCoordinate = []
        for i in range(len(listfFeIndex)):
            listFeXCoordinate.append(listiFitEvaNumByThisGen[int(listfFeIndex[i])])
        ax3.plot(listGenNum, listfAveBestIndFitnessEveryGen, '--')
        ax3.set_xticks(listfFeIndex)
        ax3.set_xticklabels(listFeXCoordinate)
        plt.show()
        print("Average Objective value:", 1/listfAveBestIndFitnessEveryGen[-1])
        print("Average Objective value:", listfAveObjValueEveryIns)


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
    listtuple_combOfInsRuns = list(itertools.product(list_ins[0:1], list_iRunsIndex))
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


def funGAMLS2_DM_ex():
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    listAllRunsSumFitEvaNumByThisGen = np.zeros((iGenNum + 1,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))
    f = open(insName, 'rb')
    listIns = []
    fig = plt.figure()
    for i in range(iInsNum):  # 8 instances
        ins = pickle.load(f)
        listIns.append(ins)
    for i in range(iActualInsNum):
        # genetic algorithm
        listfAllDiffGenBestIndFitness = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiAllDiffGenDiversityMetric1 = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiAllDiffGenDiversityMetric2 = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        listiFitEvaNumByThisGen_AllRunsAve = np.zeros((iGenNum + 1,)).tolist()
        listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve = np.zeros((iGenNum + 1,)).tolist()
        listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve = np.zeros((iGenNum + 1,)).tolist()
        for j in range(iRunsNum):  # Every instance has 10 runs experiments.
            print("Begin: ins " + str(i) + ", Runs " + str(j))
            print("Running......")
            cpuStart = time.process_time()
            # 调用GADM求解
            local_state = np.random.RandomState()
            GeneticAlgo = GAMLS2_DM.GA(listGAParameters, listIns[3], local_state)
            listdictFinalPop, listGenNum, listfBestIndFitnessEveGen, listiDiversityMetric1, listiDiversityMetric2, listiFitEvaNumByThisGen, listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd, listfEveGenProportion_belongToOnlyCurrGenLocalSearchedIndNeighbor, listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd = GeneticAlgo.funGA_main()
            cpuEnd = time.process_time()
            # 记录CPU time，累加
            listfAveCPUTimeEveryIns[i] += (cpuEnd - cpuStart)
            # 记录最终种群中最好个体的fitness和目标函数值，累加
            print("Objective value:", listdictFinalPop[0]['objectValue'])
            if listfBestIndFitnessEveGen[-1] != 1/listdictFinalPop[0]['objectValue']:
                print("Wrong. Please check funLS_DM().")
            a_2d_fEveInsEveRunObjValue[i][j] = listdictFinalPop[0]['objectValue']
            listfAveFitnessEveryIns[i] += listfBestIndFitnessEveGen[-1]
            listfAveObjValueEveryIns[i] += listdictFinalPop[0]['objectValue']
            # 为绘图准备
            new_listfBestIndFitness = [fitness * 1000 for fitness in listfBestIndFitnessEveGen]
            for g in range(len(listGenNum)):
                listfAllDiffGenBestIndFitness[g] += new_listfBestIndFitness[g]
                listiAllDiffGenDiversityMetric1[g] += listiDiversityMetric1[g]
                listiAllDiffGenDiversityMetric2[g] += listiDiversityMetric2[g]
                listiFitEvaNumByThisGen_AllRunsAve[g] += listiFitEvaNumByThisGen[g]
                listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve[g] += listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd[g]
                listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve[g] += listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd[g]

        print("End: ins " + str(i) + ", Runs " + str(j) + "\n")
        # 平均每次运行的时间
        listfAveCPUTimeEveryIns[i] /= iRunsNum
        # 平均fitness和目标函数值
        listfAveFitnessEveryIns[i] /= iRunsNum
        listfAveObjValueEveryIns[i] /= iRunsNum
        # 绘图
        listfAveBestIndFitnessEveryGen = [fitness / iRunsNum for fitness in listfAllDiffGenBestIndFitness]
        listiAveDiversityMetric1EveGen = [diversity / iRunsNum for diversity in listiAllDiffGenDiversityMetric1]
        listiAveDiversityMetric2EveGen = [diversity / iRunsNum for diversity in listiAllDiffGenDiversityMetric2]
        listiFitEvaNumByThisGen_AllRunsAve = [diversity / iRunsNum for diversity in listiFitEvaNumByThisGen_AllRunsAve]
        listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve = [diversity / iRunsNum for diversity in listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve]
        listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve = [diversity / iRunsNum for diversity in listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve]

        # fMax = np.max(listiAveDiversityMetric2EveGen)
        # fMin = np.min(listiAveDiversityMetric2EveGen)
        # for i in range(len(listiAveDiversityMetric2EveGen)):
        #     listiAveDiversityMetric2EveGen[i] = float(listiAveDiversityMetric2EveGen[i] - fMin)/(fMax - fMin)
        #     # listiAveDiversityMetric2EveGen[i] /= fMax

        ax1 = fig.add_subplot(111)
        l1, = ax1.plot(listGenNum, listfAveBestIndFitnessEveryGen, marker='*')
        ax1.set_xlabel("# of Generation")
        ax1.set_ylabel("Fitness Of Best Individual")
        # 右方Y轴
        ax2 = ax1.twinx()  # 与ax1共用1个x轴，在右方生成自己的y轴
        l2, = ax2.plot(listGenNum, listiAveDiversityMetric1EveGen, 'r',)
        # l5, = ax2.plot(listGenNum, listiAveDiversityMetric2EveGen, 'go:')
        l3, = ax2.plot(listGenNum, listfEveGenProportion_belongToLocalSearchedIndNeighbor_exceptCurrLocalSearchedInd_AllRunsAve, 'purple', marker='p', linestyle='--')
        # l4, = ax2.plot(listGenNum, listfEveGenProportion_belongToNeighborOfAllLocalSearchedInd_AllRunsAve, 'darkslategray', linestyle='-.')
        ax2.set_ylabel("Diversity Metric")
        # 上方X轴
        ax3 = ax1.twiny()  # 与ax1共用1个y轴，在上方生成自己的x轴
        ax3.set_xlabel("# of Fitness Evaluation")
        listfFeIndex = list(np.linspace(0, iGenNum, num=10+1))
        listFeXCoordinate = []
        for i in range(len(listfFeIndex)):
            listFeXCoordinate.append(listiFitEvaNumByThisGen[int(listfFeIndex[i])])
        ax3.plot(listGenNum, listfAveBestIndFitnessEveryGen, '--')
        ax3.set_xticks(listfFeIndex)
        ax3.set_xticklabels(listFeXCoordinate)
        plt.show()
        print("Average Objective value:", 1/listfAveBestIndFitnessEveryGen[-1])
        print("Average Objective value:", listfAveObjValueEveryIns)


def funGA():
    '''
    @listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha, 6:boolAllo2Faci]
    '''
    listGAParameters = [iGenNum, iPopSize, iCandidateFaciNum, fCrosRate, fMutRate, fAlpha, boolAllo2Faci]
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))
    textFile = open('E:\\VSCodeSpace\\PythonWorkspace\\Reliable-FLP\\100-node_GA_EveInsData(m=2).txt', 'a')
    plt.figure()

    f = open(insName, 'rb')
    for i in range(iInsNum):  # 10 instances
        ins = pickle.load(f)
        # print(ins.aiDemands)
        # genetic algorithm
        listfAllDiffGenBestIndFitness = np.zeros((iGenNum + 1,)).tolist()  # 算上第0代
        for j in range(iRunsNum):  # Every instance has 10 runs experiments.
            print("Begin: ins " + str(i) + ", Runs " + str(i))
            print("Running......")
            cpuStart = time.process_time()
            # 调用GA求解
            GeneticAlgo = GA.GA(listGAParameters, ins)
            finalPop, listGenNum, listfBestIndFitness = GeneticAlgo.funGA_main()
            cpuEnd = time.process_time()
            # 记录CPU time，累加
            listfAveCPUTimeEveryIns[i] += (cpuEnd - cpuStart)
            # 记录最终种群中最好个体的fitness和目标函数值，累加
            if listfBestIndFitness[-1] != 1/finalPop[0]['objectValue']:
                print("Wrong. Please check GA.")
            a_2d_fEveInsEveRunObjValue[i][j] = finalPop[0]['objectValue']
            listfAveFitnessEveryIns[i] += listfBestIndFitness[-1]
            listfAveObjValueEveryIns[i] += finalPop[0]['objectValue']
            # 为绘图准备
            new_listfBestIndFitness = [fitness * 1000 for fitness in listfBestIndFitness]
            for g in range(len(listGenNum)):
                listfAllDiffGenBestIndFitness[g] += new_listfBestIndFitness[g]
        print("End: ins " + str(i) + ", Runs " + str(j) + "\n")
        # 平均每次运行的时间
        listfAveCPUTimeEveryIns[i] /= iRunsNum
        # 平均fitness和目标函数值
        listfAveFitnessEveryIns[i] /= iRunsNum
        listfAveObjValueEveryIns[i] /= iRunsNum
        # 绘图
        listfAveBestIndFitnessEveryGen = [fitness / iRunsNum for fitness in listfAllDiffGenBestIndFitness]
        plt.plot(listGenNum, listfAveBestIndFitnessEveryGen)

    plt.xlabel("# of Generation")
    plt.ylabel("Fitness Of Best Individual (× 1e-3)")
    plt.title("Convergence Curves (100-node, m=2)")
    plt.savefig("100-node_GA_ConvergenceCurve(m=2)")
    # 将数据写入text文件
    textFile.write('\nAverage CPU time of 10 runs for each instance:\n')
    textFile.write(str(listfAveCPUTimeEveryIns))
    textFile.write('\n\nAverage fitness of 10 runs for each instance:\n')
    textFile.write(str(listfAveFitnessEveryIns))
    textFile.write('\n\nAverage objective value of 10 runs for each instance:\n')
    textFile.write(str(listfAveObjValueEveryIns))
    np.savetxt("E:\\VSCodeSpace\\PythonWorkspace\\Reliable-FLP\\100-node_GA_ObjValueEveInsEveRun(m=2).txt", a_2d_fEveInsEveRunObjValue)
    excelName = '100-node_GA_ObjValueEveInsEveRun(m=2).xls'
    funWriteExcel(excelName, a_2d_fEveInsEveRunObjValue)


def funGA_ex():
    '''
    @listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha, 6:boolAllo2Faci]
    '''
    iGenNum = 60
    iPopSize = 30
    fCrosRate = 0.9
    fMutRate = 0.1
    iInsNum = 10
    iRunsNum = 1
    boolAllo2Faci = False
    listGAParameters = [iGenNum, iPopSize, iCandidateFaciNum, fCrosRate, fMutRate, fAlpha, boolAllo2Faci]
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))
    # plt.figure()

    f = open(insName, 'rb')
    listIns = []
    for i in range(10):  # 10 instances
        ins = pickle.load(f)
        listIns.append(ins)
    for i in range(5, 6):
        # genetic algorithm
        listfAllDiffGenBestIndFitness = np.zeros((iGenNum + 1,)).tolist()  # 第0代也算上
        for j in range(iRunsNum):  # Every instance has 10 runs experiments.
            print("Begin: ins " + str(i) + ", Runs " + str(j))
            print("Running......")
            cpuStart = time.process_time()
            # 调用GA求解
            GeneticAlgo = GA.GA(listGAParameters, listIns[i])
            finalPop, listGenNum, listfBestIndFitness = GeneticAlgo.funGA_main()
            cpuEnd = time.process_time()
            print('Sol:', finalPop[0]['chromosome'])
            # 记录CPU time，累加
            listfAveCPUTimeEveryIns[i] += (cpuEnd - cpuStart)
            # 记录最终种群中最好个体的fitness和目标函数值，累加
            if listfBestIndFitness[-1] != 1/finalPop[0]['objectValue']:
                print("Wrong. Please check GA.")
            a_2d_fEveInsEveRunObjValue[i][j] = finalPop[0]['objectValue']
            listfAveFitnessEveryIns[i] += listfBestIndFitness[-1]
            listfAveObjValueEveryIns[i] += finalPop[0]['objectValue']
            # 为绘图准备
            new_listfBestIndFitness = [fitness * 1000 for fitness in listfBestIndFitness]
            for g in range(len(listGenNum)):
                listfAllDiffGenBestIndFitness[g] += new_listfBestIndFitness[g]
        print("End: ins " + str(i) + ", Runs " + str(j) + "\n")
        # 平均每次运行的时间
        print("CPU Time:")
        listfAveCPUTimeEveryIns[i] /= iRunsNum
        print(listfAveCPUTimeEveryIns[i])
        # 平均fitness和目标函数值
        listfAveFitnessEveryIns[i] /= iRunsNum
        print("Objective Value:")
        listfAveObjValueEveryIns[i] /= iRunsNum
        print(listfAveObjValueEveryIns[i])
        listfAveBestIndFitnessEveryGen = [fitness / iRunsNum for fitness in listfAllDiffGenBestIndFitness]
        # plt.plot(listGenNum, listfAveBestIndFitnessEveryGen)

    # plt.xlabel("# of Generation")
    # plt.ylabel("Fitness Of Best Individual (× 1e-3)")
    # plt.title("Convergence Curves (500-node, m=2)")
    # plt.savefig("GA_ex(m=2)")


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
    if listfBestIndFitness[-1] != 1/finalPop[0]['objectValue']:
        print("Wrong. Please check GA.")
    # 为绘图准备
    new_listfBestIndFitness = [fitness * 1000 for fitness in listfBestIndFitness]
    return cpuTime, listfBestIndFitness[-1], finalPop[0]['objectValue'], new_listfBestIndFitness


def funGA_parallel():
    '''
    @listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha, 6:boolAllo2Faci]
    '''
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))
    pool = Pool()
    list_iRunsIndex = [i for i in range(iRunsNum)]
    plotFile = open('E:\\VSCodeSpace\\PythonWorkspace\\Reliable-FLP\\ex10-node_GA_poltData(m=2).txt', 'a')
    textFile = open('E:\\VSCodeSpace\\PythonWorkspace\\Reliable-FLP\\ex10-node_GA_EveInsData(m=2).txt', 'a')
    f = open(insName, 'rb')
    list_ins = []
    for i in range(iInsNum):
        ins = pickle.load(f)
        list_ins.append(ins)
    listtuple_combOfInsRuns = list(itertools.product(list_ins, list_iRunsIndex))
    listtuple_expeResult = pool.map(funGA_single, listtuple_combOfInsRuns)
    pool.close()
    pool.join()
    if len(listtuple_expeResult) != (iInsNum * iRunsNum):
        print("Wrong. Need check.")
    plt.figure()
    for i in range(iInsNum):
        listfAllDiffGenBestIndFitness = np.zeros((iGenNum + 1,)).tolist()
        for j in range(iRunsNum):
            # 记录CPU time，累加
            listfAveCPUTimeEveryIns[i] += listtuple_expeResult[i * 10 + j][0]
            # 记录最终种群中最好个体的fitness和目标函数值，累加
            listfAveFitnessEveryIns[i] += listtuple_expeResult[i * 10 + j][1]
            listfAveObjValueEveryIns[i] += listtuple_expeResult[i * 10 + j][2]
            # 为绘图准备
            new_listfBestIndFitness = listtuple_expeResult[i * 10 + j][3]
            for g in range(len(new_listfBestIndFitness)):
                listfAllDiffGenBestIndFitness[g] += new_listfBestIndFitness[g]
            # 记录每个instance每一次run所得到的最终种群的最优个体的目标函数值
            a_2d_fEveInsEveRunObjValue[i][j] = listtuple_expeResult[i * 10 + j][2]
        # 平均每次运行的时间
        print("CPU Time:")
        listfAveCPUTimeEveryIns[i] /= iRunsNum
        print(listfAveCPUTimeEveryIns[i])
        # 平均fitness和目标函数值
        listfAveFitnessEveryIns[i] /= iRunsNum
        print("Objective Value:")
        listfAveObjValueEveryIns[i] /= iRunsNum
        print(listfAveObjValueEveryIns[i])
        # 绘图
        listfAveBestIndFitnessEveryGen = [fitness / iRunsNum for fitness in listfAllDiffGenBestIndFitness]
        plotFile.write('\nAverage best individual\'s fitness of every generation - ins: '+str(i)+'\n')
        plotFile.write(str(listfAveBestIndFitnessEveryGen))
        listGenIndex = list(np.linspace(0, iGenNum, num=(iGenNum + 1)))
        plt.plot(listGenIndex, listfAveBestIndFitnessEveryGen)

    plt.xlabel("# of Generation")
    plt.ylabel("Fitness Of Best Individual (× 1e-3)")
    plt.title("exConvergence Curves (10-node, m=2)")
    plt.savefig("ex10-node_GA_ConvergenceCurve(m=2)")
    # 将数据写入text文件
    textFile.write('\nAverage CPU time of 10 runs for each instance:\n')
    textFile.write(str(listfAveCPUTimeEveryIns))
    textFile.write('\n\nAverage fitness of 10 runs for each instance:\n')
    textFile.write(str(listfAveFitnessEveryIns))
    textFile.write('\n\nAverage objective value of 10 runs for each instance:\n')
    textFile.write(str(listfAveObjValueEveryIns))
    np.savetxt("E:\\VSCodeSpace\\PythonWorkspace\\Reliable-FLP\\ex10-node_GA_ObjValueEveInsEveRun(m=2).txt", a_2d_fEveInsEveRunObjValue)
    excelName = 'ex10-node_GA_ObjValueEveInsEveRun(m=2).xls'
    funWriteExcel(excelName, a_2d_fEveInsEveRunObjValue)


def funGA_parallel_4ins():
    iInsNum = 8
    '''
    @listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha, 6:boolAllo2Faci]
    '''
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))
    pool = Pool()
    list_iRunsIndex = [i for i in range(iRunsNum)]
    plotFile = open('E:\\VSCodeSpace\\PythonWorkspace\\Reliable-FLP\\10-node_GA_poltData(m=2).txt', 'a')
    textFile = open('E:\\VSCodeSpace\\PythonWorkspace\\Reliable-FLP\\10-node_GA_EveInsData(m=2).txt', 'a')
    f = open(insName, 'rb')
    list_ins = []
    for i in range(iInsNum):
        ins = pickle.load(f)
        list_ins.append(ins)
    # listtuple_combOfInsRuns = list(itertools.product(list_ins[:4], list_iRunsIndex))  # int(iInsNum/2) == 4, 前4个instances
    listtuple_combOfInsRuns = list(itertools.product(list_ins[4:], list_iRunsIndex))  # 后4个instances
    listtuple_expeResult = pool.map(funGA_single, listtuple_combOfInsRuns)
    pool.close()
    pool.join()
    plt.figure()
    for i in range(int(iInsNum/2)):
        listfAllDiffGenBestIndFitness = np.zeros((iGenNum + 1,)).tolist()
        for j in range(iRunsNum):
            # 记录CPU time，累加
            listfAveCPUTimeEveryIns[i] += listtuple_expeResult[i * 10 + j][0]
            # 记录最终种群中最好个体的fitness和目标函数值，累加
            listfAveFitnessEveryIns[i] += listtuple_expeResult[i * 10 + j][1]
            listfAveObjValueEveryIns[i] += listtuple_expeResult[i * 10 + j][2]
            # 为绘图准备
            new_listfBestIndFitness = listtuple_expeResult[i * 10 + j][3]
            for g in range(len(new_listfBestIndFitness)):
                listfAllDiffGenBestIndFitness[g] += new_listfBestIndFitness[g]
            # 记录每个instance每一次run所得到的最终种群的最优个体的目标函数值
            a_2d_fEveInsEveRunObjValue[i][j] = listtuple_expeResult[i * 10 + j][2]
        # 平均每次运行的时间
        print("CPU Time:")
        listfAveCPUTimeEveryIns[i] /= iRunsNum
        print(listfAveCPUTimeEveryIns[i])
        # 平均fitness和目标函数值
        listfAveFitnessEveryIns[i] /= iRunsNum
        print("Objective Value:")
        listfAveObjValueEveryIns[i] /= iRunsNum
        print(listfAveObjValueEveryIns[i])
        # 绘图
        listfAveBestIndFitnessEveryGen = [fitness / iRunsNum for fitness in listfAllDiffGenBestIndFitness]
        plotFile.write('\nAverage best individual\'s fitness of every generation - ins: '+str(i)+'\n')
        plotFile.write(str(listfAveBestIndFitnessEveryGen))
        listGenIndex = list(np.linspace(0, iGenNum, num=(iGenNum + 1)))
        plt.plot(listGenIndex, listfAveBestIndFitnessEveryGen)

    plt.xlabel("# of Generation")
    plt.ylabel("Fitness Of Best Individual (× 1e-3)")
    plt.title("Convergence Curves (10-node, m=2)")
    plt.savefig("10-node_GA_ConvergenceCurve(m=2)")
    # 将数据写入text文件
    textFile.write('\nAverage CPU time of 10 runs for each instance:\n')
    textFile.write(str(listfAveCPUTimeEveryIns))
    textFile.write('\n\nAverage fitness of 10 runs for each instance:\n')
    textFile.write(str(listfAveFitnessEveryIns))
    textFile.write('\n\nAverage objective value of 10 runs for each instance:\n')
    textFile.write(str(listfAveObjValueEveryIns))
    np.savetxt("E:\\VSCodeSpace\\PythonWorkspace\\Reliable-FLP\\10-node_GA_ObjValueEveInsEveRun(m=2).txt", a_2d_fEveInsEveRunObjValue)
    excelName = '10-node_GA_ObjValueEveInsEveRun(m=2).xls'
    funWriteExcel(excelName, a_2d_fEveInsEveRunObjValue)


def funCplex_mp():
    listCplexParameters = [iCandidateFaciNum, fAlpha]
    f = open(insName, 'rb')
    afOptimalValueEveIns = np.zeros((iInsNum,))
    listfCpuTimeEveIns = []
    textFile = open('E:\\VSCodeSpace\\PythonWorkspace\\Reliable-FLP\\100-node_Cplex_mp_data(m=2).txt', 'a')
    for i in range(iInsNum):
        print("Begin: Ins " + str(i))
        print("Running......")
        ins = pickle.load(f)
        cpu_start = time.process_time()
        cplexSolver = usecplex.CPLEX(listCplexParameters, ins)
        if boolAllo2Faci is True:
            cplexSolver.fun_fillMpModel()
        else:
            cplexSolver.fun_fillMpModel_AlloAllSelcFaci()
        sol = cplexSolver.model.solve()
        cpu_end = time.process_time()
        listfCpuTimeEveIns.append(cpu_end - cpu_start)
        afOptimalValueEveIns[i] = sol.get_objective_value()
        print(sol.solve_details)
        print("End: Ins " + str(i) + "\n")
    textFile.write('\nEvery instance\'s objective value got by Cplex-mp method:\n')
    textFile.write(str(afOptimalValueEveIns))
    textFile.write('\n\nEvery instance\'s CPU time used by Cplex-mp method:\n')
    textFile.write(str(listfCpuTimeEveIns))


def funCplex_mp_ex():
    listCplexParameters = [iCandidateFaciNum, fAlpha]
    f = open(insName, 'rb')
    listIns = []
    for i in range(iInsNum):
        listIns.append(pickle.load(f))
    afOptimalValueEveIns = np.zeros((iInsNum,))
    listfCpuTimeEveIns = []
    for i in range(5, 6):
        print("Begin: Ins " + str(i))
        print("Running......")
        ins = listIns[i]
        cpu_start = time.process_time()
        cplexSolver = usecplex.CPLEX(listCplexParameters, ins)
        if boolAllo2Faci is True:
            cplexSolver.fun_fillMpModel()
        else:
            cplexSolver.fun_fillMpModel_AlloAllSelcFaci()
        sol = cplexSolver.model.solve()
        cpu_end = time.process_time()
        listfCpuTimeEveIns.append(cpu_end - cpu_start)
        afOptimalValueEveIns[i] = sol.get_objective_value()
        # print(sol.solve_details)
        for i in range(cplexSolver.iCandidateFaciNum):
            if sol.get_value('X_'+str(i)) == 1:
                print('X_'+str(i)+" =", 1)
        print(sol.get_objective_value())
        print("End: Ins " + str(i) + "\n")


def funCplex_mp_single(fp_obInstance):
    listCplexParameters = [iCandidateFaciNum, fAlpha]
    print("Begin: Ins ")
    print("Running......")
    cpu_start = time.process_time()
    cplexSolver = usecplex.CPLEX(listCplexParameters, fp_obInstance)
    if boolAllo2Faci is True:
            cplexSolver.fun_fillMpModel()
    else:
        cplexSolver.fun_fillMpModel_AlloAllSelcFaci()
    sol = cplexSolver.model.solve()
    cpu_end = time.process_time()
    cpuTime = cpu_end - cpu_start
    fOptimalValue = sol.get_objective_value()
    print(sol.solve_details)
    print("End: Ins ")
    return cpuTime, fOptimalValue


def funCplex_mp_parallel():
    afOptimalValueEveIns = np.zeros((iInsNum,))
    listfCpuTimeEveIns = []

    f = open(insName, 'rb')
    textFile = open('E:\\VSCodeSpace\\PythonWorkspace\\Reliable-FLP\\10-node_Cplex_mp_data(m=2).txt', 'a')
    list_ins = []
    for i in range(iInsNum):
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


def funCplex_cp_single(fp_obInstance):
    listCplexParameters = [iCandidateFaciNum, fAlpha]
    print("Begin: cplex-cp")
    print("Running......")
    cpu_start = time.process_time()
    cplexSolver = usecplex.CPLEX(listCplexParameters, fp_obInstance)
    cplexSolver.fun_fillCpoModel()
    cpsol = cplexSolver.cpomodel.solve(TimeLimit=100, TimeMode='CPUTime')
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
    textFile = open('E:\\VSCodeSpace\\PythonWorkspace\\Reliable-FLP\\10-node_Cplex_cp_data(m=2).txt', 'a')
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


def funCplex_cp():
    listCplexParameters = [iCandidateFaciNum, fAlpha]
    f = open(insName, 'rb')
    afOptimalValueEveIns = np.zeros((iInsNum,))
    listfCpuTimeEveIns = []
    textFile = open('E:\\VSCodeSpace\\PythonWorkspace\\Reliable-FLP\\100-node_Cplex_cp_data(m=2).txt', 'a')
    for i in range(iInsNum):
        print("Begin: Ins " + str(i))
        print("Running......")
        ins = pickle.load(f)
        cpu_start = time.process_time()
        cplexSolver = usecplex.CPLEX(listCplexParameters, ins)
        cplexSolver.fun_fillCpoModel()
        cpsol = cplexSolver.cpomodel.solve()
        # cpsol = cplexSolver.cpomodel.solve(TimeLimit=100, TimeMode='CPUTime')
        cpu_end = time.process_time()
        listfCpuTimeEveIns.append(cpu_end - cpu_start)
        afOptimalValueEveIns[i] = cpsol.get_objective_values()[0]
        print("Solution status: " + cpsol.get_solve_status())
        print("End: Ins " + str(i) + "\n")
    textFile.write('\nEvery instance\'s objective value got by Cplex-cp method:\n')
    textFile.write(str(afOptimalValueEveIns))
    textFile.write('\n\nEvery instance\'s CPU time used by Cplex-cp method:\n')
    textFile.write(str(listfCpuTimeEveIns))


def funCplex_cp_ex():
    iInsNum = 1
    listCplexParameters = [iCandidateFaciNum, fAlpha]
    f = open(insName, 'rb')
    afOptimalValueEveIns = np.zeros((iInsNum,))
    listfCpuTimeEveIns = []
    listfGapEveIns = []
    listfBoundEveIns = []
    for i in range(iInsNum):
        print("Begin: Ins " + str(i))
        print("Running......")
        ins = pickle.load(f)
        cpu_start = time.process_time()
        cplexSolver = usecplex.CPLEX(listCplexParameters, ins)
        cplexSolver.fun_fillCpoModel()
        # cpsol = cplexSolver.cpomodel.solve()
        # cpsol = cplexSolver.cpomodel.solve(RelativeOptimalityTolerance=0.16)
        cpsol = cplexSolver.cpomodel.solve(TimeLimit=10, TimeMode='CPUTime')
        cpu_end = time.process_time()
        listfCpuTimeEveIns.append(cpu_end - cpu_start)
        afOptimalValueEveIns[i] = cpsol.get_objective_values()[0]
        listfGapEveIns.append(cpsol.get_objective_gaps()[0])
        listfBoundEveIns.append(cpsol.get_objective_bounds()[0])
        print("Solution status: " + cpsol.get_solve_status())
        print("End: Ins " + str(i) + "\n")
        print("CPU Time:", cpu_end - cpu_start)
        print("Gap:", listfGapEveIns[i])
        print("Bound:", listfBoundEveIns[i])


def funLR2():
    iMaxIterationNum = 60
    fBeta = 2.0
    fBetaMin = 1e-8
    fToleranceEpsilon = 0.0001
    boolAllo2FaciNum = True
    listLRParameters = [iMaxIterationNum, fBeta, fBetaMin, fAlpha, fToleranceEpsilon, boolAllo2FaciNum]
    listfUBEveIns = []
    listfLBEveIns = []
    f = open(insName, 'rb')
    textFile = open('ex10-node_LR2(m=2).txt', 'a')
    for i in range(iActualInsNum):
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


def funLR2_single(fp_obInstance):
    print("Begin: Ins ")
    print("Running......")
    iMaxIterationNum = 600
    fBeta = 2.0
    fBetaMin = 1e-8
    fToleranceEpsilon = 0.0001
    listLRParameters = [iMaxIterationNum, fBeta, fBetaMin, fAlpha, fToleranceEpsilon, boolAllo2Faci]
    LagRela = LR2.LagrangianRelaxation(listLRParameters, fp_obInstance)
    LagRela.funInitMultiplierLambda()
    upperBound, lowerBound = LagRela.funLR_main()
    print("End: Ins\n")
    return upperBound, lowerBound


def funLR2_parallel():
    listfUBEveIns = []
    listfLBEveIns = []
    f = open(insName, 'rb')
    textFile = open('E:\\VSCodeSpace\\PythonWorkspace\\Reliable-FLP\\100-node_LR2(m=2).txt', 'a')
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


def funLR1():
    iMaxIterationNum = 60
    fBeta = 2.0
    fBetaMin = 1e-8
    fToleranceEpsilon = 0.0001
    listLRParameters = [iMaxIterationNum, fBeta, fBetaMin, fAlpha, fToleranceEpsilon]
    listfUBEveIns = []
    listfLBEveIns = []
    f = open(insName, 'rb')
    textFile = open('E:\\VSCodeSpace\\PythonWorkspace\\Reliable-FLP\\100-node_LR1(m=2).txt', 'a')
    for i in range(iInsNum):
        ins = pickle.load(f)
        LagRela = LR1.LagrangianRelaxation(listLRParameters, ins)
        LagRela.funInitMultiplierLambda()
        upperBound, lowerBound = LagRela.funLR_main()
        listfUBEveIns.append(upperBound)
        listfLBEveIns.append(lowerBound)
    textFile.write("\nUpperbound for every instance:\n")
    textFile.write(str(listfUBEveIns))
    textFile.write("\n\nLowerbound for every instance:\n")
    textFile.write(str(listfLBEveIns))


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
    # funGAMLS2_DM_parallel()
    funLR2()
