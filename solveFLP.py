import numpy as np
import GA
import pickle
import instanceGeneration
from instanceGeneration import Instances
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import time
import xlwt


def funWriteExcel(excelName, a_2d_fEveInsEveRunObjValue):
    rowNum = a_2d_fEveInsEveRunObjValue.shape[0]
    columnNum = a_2d_fEveInsEveRunObjValue.shape[1]
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet('sheet1')
    for i in range(rowNum):
        for j in range(columnNum):
            sheet.write(i, j, a_2d_fEveInsEveRunObjValue[i][j])
    workbook.save(excelName)


def funGA():
    '''
    @listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha]
    '''
    iInsNum = 10
    iRunsNum = 10
    iGenNum = 10
    iPopSize = 10
    iCandidateFaciNum = 10
    listGAParameters = [iGenNum, iPopSize, iCandidateFaciNum, 0.9, 0.1, 1]
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))
    textFile = open('E:\\VSCodeSpace\\PythonWorkspace\\GAforFLP\\10-nodeInsExperData.txt', 'a')
    plt.figure()

    f = open('10-nodeInstances', 'rb')
    for i in range(iInsNum):  # 10 instances
        ins = pickle.load(f)
        # print(ins.aiDemands)
        # genetic algorithm
        listfAllDiffGenBestIndFitness = np.zeros((iGenNum + 1,)).tolist()  # 不算第0代
        for j in range(iRunsNum):  # Every instance has 10 runs experiments.
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
    plt.title("Convergence Curves (10-node, r=2)")
    plt.savefig("10-node")
    # 将数据写入text文件
    textFile.write('\nAverage CPU time of 10 runs for each instance:\n')
    textFile.write(str(listfAveCPUTimeEveryIns))
    textFile.write('\n\nAverage fitness of 10 runs for each instance:\n')
    textFile.write(str(listfAveFitnessEveryIns))
    textFile.write('\n\nAverage objective value of 10 runs for each instance:\n')
    textFile.write(str(listfAveObjValueEveryIns))
    np.savetxt("E:\\VSCodeSpace\\PythonWorkspace\\GAforFLP\\10-nodeObjValueEveInsEveRun.txt", a_2d_fEveInsEveRunObjValue)
    excelName = '10-nodeObjValueEveInsEveRun.xls'
    funWriteExcel(excelName, a_2d_fEveInsEveRunObjValue)


funGA()
