# -*- coding: UTF-8 -*-
from __future__ import print_function, division
from operator import itemgetter
import numpy as np
import copy
import instanceGeneration
import matplotlib.pyplot as plt
import heapq
import time


class Individal:
    def __init__(self, iIndLen, fp_local_state):
        self.aChromosome = np.zeros((iIndLen, ), dtype=np.int)
        self.fFitness = 0.0
        self.objectValue = 0.0
        for i in range(iIndLen):
            if fp_local_state.rand() <= 0.5:
                self.aChromosome[i] = 1


class GA:
    def __init__(self, listGAParameters, fp_obInstance, fp_local_state):
        '''
        Initialize parameters of GA, import instance
        @listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha, 6:boolAllo2Faci]
        '''
        self.local_state = fp_local_state
        self.iGenNum = listGAParameters[0]
        self.iPopSize = listGAParameters[1]
        self.iIndLen = listGAParameters[2]
        self.fCrosRate = listGAParameters[3]
        self.fMutRate = listGAParameters[4]
        self.fAlpha = listGAParameters[5]
        self.boolAllo2Faci = listGAParameters[6]
        self.obInstance = fp_obInstance
        if self.obInstance.iSitesNum != self.iIndLen:
            print(
                "Wrong. The number of candidate sites is not equal to the individual length. Please check."
            )

    def funInitializePop(self):
        '''
        Initialize the population.
        @return listdictInitPop
        '''
        listdictInitPop = []
        for _ in range(self.iPopSize):
            ind = Individal(self.iIndLen, self.local_state)
            listdictInitPop.append({
                'chromosome': ind.aChromosome,
                'fitness': ind.fFitness,
                'objectValue': ind.objectValue
            })
        return listdictInitPop

    def funEvaluateInd(self, fp_aChromosome):
        '''
        Note that the fitness should be the larger the better, or the method "funSelectParents" and other function which used fitness need be corrected.
        @return: fFitness
        '''
        # define fitness function according to objective function;
        w1 = 0
        w2 = 0
        if fp_aChromosome.size != self.obInstance.aiFixedCost.size:
            print("Wrong. Please make sure that the size of variable \"fp_aChromosome\" and \"self.obInstance.aiFixedCost\" equal.")
        w1 += np.dot(fp_aChromosome, self.obInstance.aiFixedCost)
        iSelcSitesNum = np.sum(fp_aChromosome)
        if iSelcSitesNum == 0:
            return 0

        for i in range(self.iIndLen):  # i represents different customers.
            aSelcSitesTransCostForI = np.multiply(
                fp_aChromosome, self.obInstance.af_2d_TransCost[i])

            aSelcSitesTransCostForI = [
                value for value in aSelcSitesTransCostForI if value != 0
            ]
            # if site i is selected, it would be missed in the above step and its trans cost is 0.
            if fp_aChromosome[i] == 1:
                aSelcSitesTransCostForI = np.append(aSelcSitesTransCostForI, 0)
            if iSelcSitesNum != len(aSelcSitesTransCostForI):
                print("Wrong in funEvaluatedInd(). Please check.")
            aSortedTransCostForI = sorted(
                aSelcSitesTransCostForI)  # ascending order

            # w1 += self.obInstance.aiDemands[i] * aSortedTransCostForI[0]

            # j represents the facilities that allocated to the customer i
            iAlloFaciNum = 2 if self.boolAllo2Faci is True else len(aSortedTransCostForI)
            for j in range(iAlloFaciNum):
                p = self.obInstance.fFaciFailProb
                w2 += self.obInstance.aiDemands[i] * aSortedTransCostForI[
                    j] * pow(p, j) * (1 - p)

        # The larger, the better.
        fObjectValue = w1 + self.fAlpha * w2
        fFitness = 1 / (w1 + self.fAlpha * w2)
        return fFitness, fObjectValue

    def funEvaluatePop(self, fp_listdictPop):
        '''
        This method is used to evaluate the population.
        @return: listdictPopBefSurv
        '''
        for i in range(len(fp_listdictPop)):
            # make sure that at least 2 facilities are established to garantee reliability.
            if sum(fp_listdictPop[i]['chromosome']) < 2:
                self.funModifyInd(fp_listdictPop[i]['chromosome'])

            fp_listdictPop[i]['fitness'], fp_listdictPop[i]['objectValue'] = self.funEvaluateInd(
                fp_listdictPop[i]['chromosome'])
        return fp_listdictPop

    def funModifyInd(self, fp_aChromosome):
        '''
        At least 2 facilitis are established to garantee the reliable.
        The modify of "fp_aChromosome" will influent the real aChromosome.
        '''
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(fp_aChromosome)
        iRealFaciNum = sum(fp_aChromosome)
        aSortedFixedCost = sorted(self.obInstance.aiFixedCost)  # default ascending order
        while iRealFaciNum < 2:
            for j in range(self.iIndLen):
                if fp_aChromosome[np.where(self.obInstance.aiFixedCost == aSortedFixedCost[j])[0][0]] == 1:
                    print(np.where(self.obInstance.aiFixedCost==aSortedFixedCost[j])[0][0])
                    print("###################################################")
                    print(fp_aChromosome)
                    continue
                else:
                    fp_aChromosome[np.where(self.obInstance.aiFixedCost == aSortedFixedCost[j])[0][0]] = 1
                    iRealFaciNum += 1
                    if iRealFaciNum >= 2:
                        break
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    def funSelectParents(self, fp_listdictCurrPop, fp_iIndIndex=None):
        '''
        Roulte wheel method to choose parents.
        If the value of "fp_iIndIndex" is None, choose both 2 parents by roulte wheel method.
        If "fp_iIndIndex" is an integer, i.e., individual index, choose only one parent by roulte wheel method. The other one is the individual whose index is "fp_iIndIndex".
        Note that our fitness value is the larger the better.
        @return: listdictParents
        '''
        listdictParents = []
        fFitnessSum = sum(ind['fitness'] for ind in fp_listdictCurrPop)
        fProb = [
            fp_listdictCurrPop[i]['fitness'] / fFitnessSum
            for i in range(len(fp_listdictCurrPop))
        ]
        if fp_iIndIndex is None:
            adictParents = self.local_state.choice(fp_listdictCurrPop,
                                            size=2,
                                            p=fProb)
            listdictParents.extend((adictParents[0], adictParents[1]))
        else:
            listdictParents.extend(
                (
                    fp_listdictCurrPop[fp_iIndIndex],
                    self.local_state.choice(fp_listdictCurrPop, size=1, p=fProb)[
                        0
                    ],
                )
            )
        # 根据实验，listdictParents跟fp_listdictCurrPop是一体的，改变一个会影响另外一个
        return listdictParents

    def funCrossover(self,
                     fp_listdictCurrPop,
                     fp_fCrosRate,
                     fp_iHowSelPare=None):
        '''
        The value of formal parameter "fp_iHowSelPare" determines how to choose parents. If fp_iHowSelPare==1, the "fp_iIndIndex" of "funSelectParents" should be set to "None" and choose two parents from the population according to roulette wheel.
        Otherwise only choose one parent according to roulette wheel.
        return: listdictPopAfCros
        '''
        if len(fp_listdictCurrPop) != self.iPopSize:
            print(
                "Someting wrong. The population size before crossover is abnormal."
            )

        listdictPopAfCros = []
        for i in range(len(fp_listdictCurrPop)):
            if self.local_state.rand() < fp_fCrosRate:
                aOffs1 = np.zeros((self.iIndLen, ), dtype=np.int)
                aOffs2 = np.zeros((self.iIndLen, ), dtype=np.int)
                listdictParents = []
                if fp_iHowSelPare == 1:
                    listdictParents = self.funSelectParents(fp_listdictCurrPop)
                else:
                    listdictParents = self.funSelectParents(fp_listdictCurrPop,
                                                            fp_iIndIndex=i)
                # One-point crossover
                crossPoint = self.local_state.randint(1, self.iIndLen)
                for j in range(self.iIndLen):
                    if j < crossPoint:
                        aOffs1[j] = listdictParents[0]['chromosome'][j]
                        aOffs2[j] = listdictParents[1]['chromosome'][j]
                    else:
                        aOffs1[j] = listdictParents[1]['chromosome'][j]
                        aOffs2[j] = listdictParents[0]['chromosome'][j]
                listdictPopAfCros.extend(
                    (
                        {'chromosome': aOffs1, 'fitness': 0.0, 'objectValue': 0.0},
                        {'chromosome': aOffs2, 'fitness': 0.0, 'objectValue': 0.0},
                    )
                )
        # "listdictPopAfCros" has no relation to "fp_listdictCurrPop"
        return listdictPopAfCros

    def funMutation(self, fp_listdictPopAfCros):
        '''
        @return: listdictPopAfMuta
        '''
        for i in range(len(fp_listdictPopAfCros)):
            for j in range(self.iIndLen):
                # For every individual's every gene, determining whether mutating
                if self.local_state.rand() < self.fMutRate:
                    fp_listdictPopAfCros[i]['chromosome'][j] = (
                        fp_listdictPopAfCros[i]['chromosome'][j] + 1) % 2
        return self.funEvaluatePop(fp_listdictPopAfCros)

    def funSurvival(self, fp_listdictCurrPop, fp_listdictPopAfMuta):
        '''
        @fp_listdictCurrPop: current population
        @fp_listdictPopAfMuta: population after crossover and mutation
        @return: fp_listdictCurrPop
        survival strategy: (μ+λ) strategy
        '''
        # combine the current pop and after-mutation-pop, and overwrite the current pop
        fp_listdictCurrPop.extend(fp_listdictPopAfMuta)
        # descending sort
        fp_listdictCurrPop.sort(key=itemgetter('fitness'), reverse=True)
        # 对前10个个体，构造邻域种群并进行评价
        listdictNeighborPopAfEva = self.funLocalNeighborhood(fp_listdictCurrPop)
        # 将当前种群与前10个个体的邻域种群合并
        fp_listdictCurrPop.extend(listdictNeighborPopAfEva)
        # 降序排列
        fp_listdictCurrPop.sort(key=itemgetter('fitness'), reverse=True)
        # only the first μ can survive
        fp_listdictCurrPop = fp_listdictCurrPop[:self.iPopSize]
        return fp_listdictCurrPop

    def funLocalNeighborhood(self, fp_listdictCurrPop):
        '''
        Use local search to the best 10 individuals
        '''
        listdictNeighborPop = []
        for i in range(10):  # Do local search process for the best 10 individuals
            for j in range(self.iIndLen):
                dictInd = copy.deepcopy(fp_listdictCurrPop[i])
                dictInd['chromosome'][j] = (dictInd['chromosome'][j] + 1) % 2
                dictInd['fitness'] = 0
                dictInd['objectValue'] = 0
                listdictNeighborPop.append(dictInd)
        return self.funEvaluatePop(listdictNeighborPop)

    def funGA_main(self):
        '''
        The main process of genetic algorithm.
        @return listdictFinalPop
        '''
        listdictInitPop = self.funInitializePop()
        # By this time, both CurrPop and InitPop point to the same variable.
        listdictCurrPop = self.funEvaluatePop(listdictInitPop)
        listdictCurrPop = copy.deepcopy(listdictInitPop)
        listdictBestInd = heapq.nlargest(1, listdictInitPop, key=lambda x: x['fitness'])
        listfBestIndFitness = [listdictBestInd[0]['fitness']]
        for gen in range(self.iGenNum):
            print("Gen:", gen)
            listdictPopAfCros = self.funCrossover(listdictCurrPop,
                                                  self.fCrosRate)
            listdictPopAfMuta = self.funMutation(listdictPopAfCros)
            listdictCurrPop = self.funSurvival(listdictCurrPop,
                                               listdictPopAfMuta)
            listfBestIndFitness.append(listdictCurrPop[0]['fitness'])
        listdictFinalPop = listdictCurrPop
        # plot figure
        listGenIndex = list(np.linspace(0, self.iGenNum, num=(self.iGenNum + 1)))
        # plt.figure()
        # plt.plot(genNum, listfBestIndFitness)
        # plt.xlabel("# of Generation")
        # plt.ylabel("Fitness Of Best Individual")
        # plt.savefig("line.jpg")
        return listdictFinalPop, listGenIndex, listfBestIndFitness


if __name__ == '__main__':
    '''
    Test the genetic algorithm.

    listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha, 6:boolAllo2Faci]

    listInstPara=[0:iSitesNum, 1:iScenNum, 2:iDemandLB, 3:iDemandUB, 4:iFixedCostLB, 5:iFixedCostUP, 6:iCoordinateLB, 7:iCoordinateUB, 8:fFaciFailProb]

    The value of  2:iIndLen and 0:iSitesNum should be equal.
    '''
    iGenNum = 100
    iPopSize = 200
    iCandidateFaciNum = 50
    fCrosRate = 0.9
    fMutRate = 0.1
    fAlpha = 1
    boolAllo2Faci = True
    listGAParameters = [iGenNum, iPopSize, iCandidateFaciNum, fCrosRate, fMutRate, fAlpha, boolAllo2Faci]
    listInstPara = [iCandidateFaciNum, 1, 0, 1000, 500, 1500, 0, 1, 0.05]
    start = time.process_time()
    # generate instance
    obInstance = instanceGeneration.Instances(listInstPara)
    obInstance.funGenerateInstances()
    # genetic algorithm
    geneticAlgo = GA(listGAParameters, obInstance)
    listdictFinalPop, listGenNum, listfBestIndFitnessEveGen = geneticAlgo.funGA_main()
    end = time.process_time()
    plt.figure()
    plt.plot(listGenNum, listfBestIndFitnessEveGen)
    plt.xlabel("# of Generation")
    plt.ylabel("Fitness Of Best Individual")
    plt.show()
    # plt.savefig("line.jpg")
    print("CPU Time:", end - start)
