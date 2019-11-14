import numpy as np
import GA
import usecplex
import pickle
import instanceGeneration
from instanceGeneration import Instances
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import time
import xlwt
import LR1
import LR2

# Global variables
iInsNum = 10
iRunsNum = 10
fAlpha = 1.0
iCandidateFaciNum = 50
insName = '50-nodeInstances'


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
    iGenNum = 100
    iPopSize = 200
    fCrosRate = 0.9
    fMutRate = 0.1
    listGAParameters = [iGenNum, iPopSize, iCandidateFaciNum, fCrosRate, fMutRate, fAlpha]
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))
    textFile = open('E:\\VSCodeSpace\\PythonWorkspace\\GAforFLP\\50-node_GA_EveInsData(m=2).txt', 'a')
    plt.figure()

    f = open(insName, 'rb')
    for i in range(iInsNum):  # 10 instances
        ins = pickle.load(f)
        # print(ins.aiDemands)
        # genetic algorithm
        listfAllDiffGenBestIndFitness = np.zeros((iGenNum + 1,)).tolist()  # 不算第0代
        for j in range(iRunsNum):  # Every instance has 10 runs experiments.
            print("Begin: ins " + i + ", Runs " + j)
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
        print("End: ins " + i + ", Runs " + j)
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
    plt.title("Convergence Curves (50-node, m=2)")
    plt.savefig("50-node_GA_ConvergenceCurve(m=2)")
    # 将数据写入text文件
    textFile.write('\nAverage CPU time of 10 runs for each instance:\n')
    textFile.write(str(listfAveCPUTimeEveryIns))
    textFile.write('\n\nAverage fitness of 10 runs for each instance:\n')
    textFile.write(str(listfAveFitnessEveryIns))
    textFile.write('\n\nAverage objective value of 10 runs for each instance:\n')
    textFile.write(str(listfAveObjValueEveryIns))
    np.savetxt("E:\\VSCodeSpace\\PythonWorkspace\\GAforFLP\\50-node_GA_ObjValueEveInsEveRun(m=2).txt", a_2d_fEveInsEveRunObjValue)
    excelName = '50-node_GA_ObjValueEveInsEveRun(m=2).xls'
    funWriteExcel(excelName, a_2d_fEveInsEveRunObjValue)


def funCplex_mp():
    listCplexParameters = [iCandidateFaciNum, fAlpha]
    f = open(insName, 'rb')
    afOptimalValueEveIns = np.zeros((iInsNum,))
    listfCpuTimeEveIns = []
    textFile = open('E:\\VSCodeSpace\\PythonWorkspace\\GAforFLP\\50-node_Cplex_mp_data(m=2).txt', 'a')
    for i in range(iInsNum):
        ins = pickle.load(f)
        cpu_start = time.process_time()
        cplexSolver = usecplex.CPLEX(listCplexParameters, ins)
        cplexSolver.fun_fillMpModel()
        sol = cplexSolver.model.solve()
        cpu_end = time.process_time()
        listfCpuTimeEveIns.append(cpu_end - cpu_start)
        afOptimalValueEveIns[i] = sol.get_objective_value()
        print(sol.solve_details)
    textFile.write('\nEvery instance\'s objective value got by Cplex-mp method:\n')
    textFile.write(str(afOptimalValueEveIns))
    textFile.write('\n\nEvery instance\'s CPU time used by Cplex-mp method:\n')
    textFile.write(str(listfCpuTimeEveIns))


def funLR2():
    iMaxIterationNum = 60
    fBeta = 2.0
    fBetaMin = 1e-8
    fToleranceEpsilon = 0.0001
    listLRParameters = [iMaxIterationNum, fBeta, fBetaMin, fAlpha, fToleranceEpsilon]
    listfUBEveIns = []
    listfLBEveIns = []
    f = open(insName, 'rb')
    textFile = open('E:\\VSCodeSpace\\PythonWorkspace\\GAforFLP\\50-node_LR2(m=2).txt', 'a')
    for i in range(iInsNum):
        print("Begin: Ins " + i)
        print("Running......")
        ins = pickle.load(f)
        LagRela = LR2.LagrangianRelaxation(listLRParameters, ins)
        LagRela.funInitMultiplierLambda()
        upperBound, lowerBound = LagRela.funLR_main()
        listfUBEveIns.append(upperBound)
        listfLBEveIns.append(lowerBound)
        print("End: Ins " + i)
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
    textFile = open('E:\\VSCodeSpace\\PythonWorkspace\\GAforFLP\\50-node_LR1(m=2).txt', 'a')
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


# funGA()
# funCplex_mp()
funLR2()
# funLR1()
