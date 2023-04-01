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
iCandidateFaciNum = 1000
insName = '1000-nodeInstances'

'''
@listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha]
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


def funGA():
    '''
    @listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha]
    '''
    iGenNum = 150
    iPopSize = 200
    fCrosRate = 0.9
    fMutRate = 0.1
    listGAParameters = [iGenNum, iPopSize, iCandidateFaciNum, fCrosRate, fMutRate, fAlpha]
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))
    textFile = open('E:\\VSCodeSpace\\PythonWorkspace\\GAforFLP\\100-node_GA_EveInsData(m=2).txt', 'a')
    plt.figure()

    f = open(insName, 'rb')
    for i in range(iInsNum):  # 10 instances
        ins = pickle.load(f)
        # print(ins.aiDemands)
        # genetic algorithm
        listfAllDiffGenBestIndFitness = np.zeros((iGenNum + 1,)).tolist()  # 不算第0代
        for j in range(iRunsNum):  # Every instance has 10 runs experiments.
            print(f"Begin: ins {str(i)}, Runs {str(i)}")
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
        print(f"End: ins {str(i)}, Runs {str(j)}" + "\n")
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
    np.savetxt("E:\\VSCodeSpace\\PythonWorkspace\\GAforFLP\\100-node_GA_ObjValueEveInsEveRun(m=2).txt", a_2d_fEveInsEveRunObjValue)
    excelName = '100-node_GA_ObjValueEveInsEveRun(m=2).xls'
    funWriteExcel(excelName, a_2d_fEveInsEveRunObjValue)


def funGA_ex():
    '''
    @listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha]
    '''
    iGenNum = 150
    iPopSize = 200
    fCrosRate = 0.9
    fMutRate = 0.1
    iInsNum = 1
    iRunsNum = 1
    listGAParameters = [iGenNum, iPopSize, iCandidateFaciNum, fCrosRate, fMutRate, fAlpha]
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))
    plt.figure()

    f = open(insName, 'rb')
    for i in range(iInsNum):  # 10 instances
        ins = pickle.load(f)
        # print(ins.aiDemands)
        # genetic algorithm
        listfAllDiffGenBestIndFitness = np.zeros((iGenNum + 1,)).tolist()  # 不算第0代
        for j in range(iRunsNum):  # Every instance has 10 runs experiments.
            print(f"Begin: ins {str(i)}, Runs {str(j)}")
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
        print(f"End: ins {str(i)}, Runs {str(j)}" + "\n")
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
        plt.plot(listGenNum, listfAveBestIndFitnessEveryGen)

    plt.xlabel("# of Generation")
    plt.ylabel("Fitness Of Best Individual (× 1e-3)")
    plt.title("Convergence Curves (500-node, m=2)")
    plt.savefig("GA_ex(m=2)")


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
    # 为绘图准备
    new_listfBestIndFitness = [fitness * 1000 for fitness in listfBestIndFitness]
    return cpuTime, listfBestIndFitness[-1], finalPop[0]['objectValue'], new_listfBestIndFitness


def funGA_parallel():
    '''
    @listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha]
    '''
    listfAveFitnessEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveObjValueEveryIns = np.zeros((iInsNum,)).tolist()
    listfAveCPUTimeEveryIns = np.zeros((iInsNum,)).tolist()
    a_2d_fEveInsEveRunObjValue = np.zeros((iInsNum, iRunsNum))
    pool = Pool()
    list_iRunsIndex = list(range(iRunsNum))
    textFile = open('/home/zhanghan/pythonworkspace/reliableFLP/LinuxExperimentsResults/1000-node/1000-node_GA_EveInsData(m=2).txt', 'a')
    f = open(insName, 'rb')
    list_ins = []
    for _ in range(iInsNum):
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
        listGenIndex = list(np.linspace(0, iGenNum, num=(iGenNum + 1)))
        plt.plot(listGenIndex, listfAveBestIndFitnessEveryGen)

    plt.xlabel("# of Generation")
    plt.ylabel("Fitness Of Best Individual (× 1e-3)")
    plt.title("Convergence Curves (1000-node, m=2)")
    plt.savefig("1000-node_GA_ConvergenceCurve(m=2)")
    # 将数据写入text文件
    textFile.write('\nAverage CPU time of 10 runs for each instance:\n')
    textFile.write(str(listfAveCPUTimeEveryIns))
    textFile.write('\n\nAverage fitness of 10 runs for each instance:\n')
    textFile.write(str(listfAveFitnessEveryIns))
    textFile.write('\n\nAverage objective value of 10 runs for each instance:\n')
    textFile.write(str(listfAveObjValueEveryIns))
    np.savetxt("/home/zhanghan/pythonworkspace/reliableFLP/LinuxExperimentsResults/1000-node/1000-node_GA_ObjValueEveInsEveRun(m=2).txt", a_2d_fEveInsEveRunObjValue)
    excelName = '1000-node_GA_ObjValueEveInsEveRun(m=2).xls'
    funWriteExcel(excelName, a_2d_fEveInsEveRunObjValue)


def funCplex_mp():
    listCplexParameters = [iCandidateFaciNum, fAlpha]
    f = open(insName, 'rb')
    afOptimalValueEveIns = np.zeros((iInsNum,))
    listfCpuTimeEveIns = []
    textFile = open('E:\\VSCodeSpace\\PythonWorkspace\\GAforFLP\\100-node_Cplex_mp_data(m=2).txt', 'a')
    for i in range(iInsNum):
        print(f"Begin: Ins {str(i)}")
        print("Running......")
        ins = pickle.load(f)
        cpu_start = time.process_time()
        cplexSolver = usecplex.CPLEX(listCplexParameters, ins)
        cplexSolver.fun_fillMpModel()
        sol = cplexSolver.model.solve()
        cpu_end = time.process_time()
        listfCpuTimeEveIns.append(cpu_end - cpu_start)
        afOptimalValueEveIns[i] = sol.get_objective_value()
        print(sol.solve_details)
        print(f"End: Ins {str(i)}" + "\n")
    textFile.write('\nEvery instance\'s objective value got by Cplex-mp method:\n')
    textFile.write(str(afOptimalValueEveIns))
    textFile.write('\n\nEvery instance\'s CPU time used by Cplex-mp method:\n')
    textFile.write(str(listfCpuTimeEveIns))


def funCplex_cp():
    listCplexParameters = [iCandidateFaciNum, fAlpha]
    f = open(insName, 'rb')
    afOptimalValueEveIns = np.zeros((iInsNum,))
    listfCpuTimeEveIns = []
    textFile = open('E:\\VSCodeSpace\\PythonWorkspace\\GAforFLP\\100-node_Cplex_cp_data(m=2).txt', 'a')
    for i in range(iInsNum):
        print(f"Begin: Ins {str(i)}")
        print("Running......")
        ins = pickle.load(f)
        cpu_start = time.process_time()
        cplexSolver = usecplex.CPLEX(listCplexParameters, ins)
        cplexSolver.fun_fillCpoModel()
        cpsol = cplexSolver.cpomodel.solve(TimeLimit=100)
        cpu_end = time.process_time()
        listfCpuTimeEveIns.append(cpu_end - cpu_start)
        afOptimalValueEveIns[i] = cpsol.get_objective_values()[0]
        print(f"Solution status: {cpsol.get_solve_status()}")
        print(f"End: Ins {str(i)}" + "\n")
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
    for i in range(iInsNum):
        print(f"Begin: Ins {str(i)}")
        print("Running......")
        ins = pickle.load(f)
        cpu_start = time.process_time()
        cplexSolver = usecplex.CPLEX(listCplexParameters, ins)
        cplexSolver.fun_fillCpoModel()
        cpsol = cplexSolver.cpomodel.solve(TimeLimit=100)
        cpu_end = time.process_time()
        listfCpuTimeEveIns.append(cpu_end - cpu_start)
        afOptimalValueEveIns[i] = cpsol.get_objective_values()[0]
        print(f"Solution status: {cpsol.get_solve_status()}")
        print(f"End: Ins {str(i)}" + "\n")
        print("CPU Time:", cpu_end - cpu_start)


def funLR2():
    iMaxIterationNum = 60
    fBeta = 2.0
    fBetaMin = 1e-8
    fToleranceEpsilon = 0.0001
    listLRParameters = [iMaxIterationNum, fBeta, fBetaMin, fAlpha, fToleranceEpsilon]
    listfUBEveIns = []
    listfLBEveIns = []
    f = open(insName, 'rb')
    textFile = open('E:\\VSCodeSpace\\PythonWorkspace\\GAforFLP\\100-node_LR2(m=2).txt', 'a')
    for i in range(iInsNum):
        print(f"Begin: Ins {str(i)}")
        print("Running......")
        ins = pickle.load(f)
        LagRela = LR2.LagrangianRelaxation(listLRParameters, ins)
        LagRela.funInitMultiplierLambda()
        upperBound, lowerBound = LagRela.funLR_main()
        listfUBEveIns.append(upperBound)
        listfLBEveIns.append(lowerBound)
        print(f"End: Ins {str(i)}" + "\n")
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
    textFile = open('E:\\VSCodeSpace\\PythonWorkspace\\GAforFLP\\100-node_LR1(m=2).txt', 'a')
    for _ in range(iInsNum):
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
# funLR2()
# funLR1()
# funGA_ex()
# funCplex_cp_ex()
if __name__ == '__main__':
    funGA_parallel()
