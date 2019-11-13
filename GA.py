from __future__ import print_function, division
from operator import itemgetter
import numpy as np
import copy
import instanceGeneration
import matplotlib.pyplot as plt
import heapq


class Individal:
    def __init__(self, iIndLen):
        self.aChromosome = np.zeros((iIndLen, ), dtype=np.int)
        self.fFitness = 0.0
        self.objectValue = 0.0
        for i in range(iIndLen):
            if np.random.rand() <= 0.5:
                self.aChromosome[i] = 1


class GA:
    def __init__(self, listGAParameters, fp_obInstance):
        '''
        Initialize parameters of GA, import instance
        @listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha]
        '''
        self.iGenNum = listGAParameters[0]
        self.iPopSize = listGAParameters[1]
        self.iIndLen = listGAParameters[2]
        self.fCrosRate = listGAParameters[3]
        self.fMutRate = listGAParameters[4]
        self.fAlpha = listGAParameters[5]
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
        for i in range(self.iPopSize):
            ind = Individal(self.iIndLen)
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
                value for (index, value) in enumerate(aSelcSitesTransCostForI)
                if value != 0
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
            # for j in range(len(aSortedTransCostForI)): # 把所有Xj=1的点都分给i
            for j in range(2):  # 每个i只有两个级别的供应点
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
        listdictPopAfEval = fp_listdictPop
        return listdictPopAfEval

    def funModifyInd(self, fp_aChromosome):
        '''
        At least 2 facilitis are established to garantee the reliable.
        The modify of "fp_aChromosome" will influent the real aChromosome.
        '''
        iRealFaciNum = sum(fp_aChromosome)
        aSortedFixedCost = sorted(self.obInstance.aiFixedCost)  # default ascending order
        while iRealFaciNum < 2:
            for j in range(2):
                if fp_aChromosome[np.where(self.obInstance.aiFixedCost == aSortedFixedCost[j])[0][0]] == 1:
                    continue
                else:
                    fp_aChromosome[np.where(self.obInstance.aiFixedCost == aSortedFixedCost[j])[0][0]] = 1
                    iRealFaciNum += 1

    def funSelectParents(self, fp_listdictCurrPop, fp_iIndIndex=None):
        '''
        Roulte wheel method to choose parents.
        If the value of "fp_iIndIndex" is None, choose both 2 parents by roulte wheel method.
        If "fp_iIndIndex" is an integer, i.e., individual index, choose only one parent by roulte wheel method. The other one is the individual whose index is "fp_iIndIndex".
        Note that our fitness value is the larger the better.
        @return: listdictParents
        '''
        fProb = []
        listdictParents = []
        fFitnessSum = sum(ind['fitness'] for ind in fp_listdictCurrPop)
        for i in range(len(fp_listdictCurrPop)):
            fProb.append(fp_listdictCurrPop[i]['fitness'] / fFitnessSum)
        if fp_iIndIndex is None:
            adictParents = np.random.choice(fp_listdictCurrPop,
                                            size=2,
                                            p=fProb)
            listdictParents.append(adictParents[0])
            listdictParents.append(adictParents[1])
        else:
            listdictParents.append(fp_listdictCurrPop[fp_iIndIndex])
            listdictParents.append(
                np.random.choice(fp_listdictCurrPop, size=1, p=fProb)[0])
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
            if np.random.rand() < fp_fCrosRate:
                aOffs1 = np.zeros((self.iIndLen, ), dtype=np.int)
                aOffs2 = np.zeros((self.iIndLen, ), dtype=np.int)
                listdictParents = []
                if fp_iHowSelPare == 1:
                    listdictParents = self.funSelectParents(fp_listdictCurrPop)
                else:
                    listdictParents = self.funSelectParents(fp_listdictCurrPop,
                                                            fp_iIndIndex=i)
                # One-point crossover
                crossPoint = np.random.randint(1, self.iIndLen)
                for j in range(self.iIndLen):
                    if j < crossPoint:
                        aOffs1[j] = listdictParents[0]['chromosome'][j]
                        aOffs2[j] = listdictParents[1]['chromosome'][j]
                    else:
                        aOffs1[j] = listdictParents[1]['chromosome'][j]
                        aOffs2[j] = listdictParents[0]['chromosome'][j]
                listdictPopAfCros.append({
                    'chromosome': aOffs1,
                    'fitness': 0.0,
                    'objectValue': 0.0
                })
                listdictPopAfCros.append({
                    'chromosome': aOffs2,
                    'fitness': 0.0,
                    'objectValue': 0.0
                })
        # "listdictPopAfCros" has no relation to "fp_listdictCurrPop"
        return listdictPopAfCros

    def funMutation(self, fp_listdictPopAfCros):
        '''
        @return: listdictPopAfMuta
        '''
        for i in range(len(fp_listdictPopAfCros)):
            for j in range(self.iIndLen):
                # For every individual's every gene, determining whether mutating
                if np.random.rand() < self.fMutRate:
                    fp_listdictPopAfCros[i]['chromosome'][j] = (
                        fp_listdictPopAfCros[i]['chromosome'][j] + 1) % 2
        listdictPopAfMuta = self.funEvaluatePop(
            fp_listdictPopAfCros)  # evaluate population
        # listdictPopAfMuta is the same one as fp_listdictPopAfCros
        return listdictPopAfMuta

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
        # only the first μ can survive
        fp_listdictCurrPop = fp_listdictCurrPop[:self.iPopSize]
        return fp_listdictCurrPop

    def funGA_main(self):
        '''
        The main process of genetic algorithm.
        @return listdictFinalPop
        '''
        listdictInitPop = self.funInitializePop()
        # By this time, both CurrPop and InitPop point to the same variable.
        listdictCurrPop = self.funEvaluatePop(listdictInitPop)
        listdictCurrPop = copy.deepcopy(listdictInitPop)
        # record the fitness of the best individual for every generation
        listfBestIndFitness = []
        listdictBestInd = heapq.nlargest(1, listdictInitPop, key=lambda x: x['fitness'])
        listfBestIndFitness.append(listdictBestInd[0]['fitness'])
        for gen in range(self.iGenNum):
            listdictPopAfCros = self.funCrossover(listdictCurrPop,
                                                  self.fCrosRate)
            listdictPopAfMuta = self.funMutation(listdictPopAfCros)
            listdictCurrPop = self.funSurvival(listdictCurrPop,
                                               listdictPopAfMuta)
            listfBestIndFitness.append(listdictCurrPop[0]['fitness'])
        listdictFinalPop = listdictCurrPop
        # plot figure
        genNum = list(np.linspace(0, self.iGenNum, num=(self.iGenNum + 1)))
        # plt.figure()
        # plt.plot(genNum, listfBestIndFitness)
        # plt.xlabel("# of Generation")
        # plt.ylabel("Fitness Of Best Individual")
        # plt.savefig("line.jpg")
        return listdictFinalPop, genNum, listfBestIndFitness


if __name__ == '__main__':
    '''
    Test the genetic algorithm.

    listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha]

    listInstPara=[0:iSitesNum, 1:iScenNum, 2:iDemandLB, 3:iDemandUB, 4:iFixedCostLB, 5:iFixedCostUP, 6:iCoordinateLB, 7:iCoordinateUB, 8:fFaciFailProb]

    The value of  2:iIndLen and 0:iSitesNum should be equal.
    '''
    listGAParameters = [10, 10, 10, 0.9, 0.1, 1]
    listInstPara = [10, 1, 0, 1000, 500, 1500, 0, 1, 0.05]
    # generate instance
    obInstance = instanceGeneration.Instances(listInstPara)
    obInstance.funGenerateInstances()
    # genetic algorithm
    geneticAlgo = GA(listGAParameters, obInstance)
    finalPop = geneticAlgo.funGA_main()
    print(finalPop)
