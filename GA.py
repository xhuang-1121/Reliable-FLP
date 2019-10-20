from __future__ import print_function, division
import random
import math
from operator import itemgetter
import numpy as np
import copy


class Individal:
    def __init__(self, iIndLen):
        self.aChromosome = np.zeros((iIndLen, ), dtype=np.int)
        self.fFitness = 0.0
        for i in range(iIndLen):
            if np.random.rand() <= 0.5:
                self.aChromosome[i] = 1


class GA:
    def __init__(self, listParameters):
        # listParameters = [iGenNum, iPopSize, iIndLen, fCrosRate, fMutRate]
        self.iGenNum = listParameters[0]
        self.iPopSize = listParameters[1]
        self.iIndLen = listParameters[2]
        self.fCrosRate = listParameters[3]
        self.fMutRate = listParameters[4]

    def funInitializePop(self):
        listdictInitPop = []
        for i in range(self.iPopSize):
            ind = Individal(self.iIndLen)
            listdictInitPop.append({
                'chromosome': ind.aChromosome,
                'fitness': ind.fFitness
            })
        return listdictInitPop

    def funEvaluateInd(self, fp_aChromosome):
        '''
        Note that the fitness should be the larger the better,
        or the method "funSelectParents" and other function which
        used fitness need be corrected.
        '''
        # TODO 根据目标函数定义fitness function;
        fFitness = 0
        return fFitness

    def funEvaluatePop(self, fp_listdictPop):
        '''
        该函数用于评价种群；
        return: listdictPopBefSurv
        '''
        for i in range(len(fp_listdictPop)):
            fp_listdictPop[i]['fitness'] = self.funEvaluateInd(
                fp_listdictPop[i]['chromosome'])
        listdictPopBefSurv = fp_listdictPop
        return listdictPopBefSurv

    def funSelectParents(self, fp_listdictCurrPop, fp_iIndIndex=None):
        '''
        轮盘赌方法选择交叉用的两个个体；
        这里有两种方法，一是每次都选出两个个体进行交叉；
        二是对每个个体，都给它再从种群中选一个进行交叉；
        When the value of "fp_iIndIndex" is None, choose the first approach,
        otherwise choose the second approach;
        Note that our fitness value is the larger the better.
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

    def funCrossover(self, fp_listdictCurrPop, fp_fCrosRate, fp_iHowSelPare):
        '''
        The value of formal parameter "fp_iSelPare" determines how to choose parents. If fp_iSelPare==1, the "fp_iIndIndex" of "funSelectParents" should be set to "None" and choose two parents from the population according to roulette wheel.
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
                    'fitness': 0.0
                })
                listdictPopAfCros.append({
                    'chromosome': aOffs2,
                    'fitness': 0.0
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
        parameters: current population and population after crossover and mutation
        survival strategy: (μ+λ) strategy
        '''
        # combine the current pop and after-mutation-pop, and overwrite the current pop
        fp_listdictCurrPop.extend(fp_listdictPopAfMuta)
        # ascending sort
        fp_listdictCurrPop.sort(key=itemgetter('fitness'), reverse=True)
        # only the first μ can survive
        fp_listdictCurrPop = fp_listdictCurrPop[:self.iPopSize]
        return fp_listdictCurrPop

    def funGA_main(self):
        listdictInitPop = self.funInitializePop()
        listdictCurrPop = self.funEvaluatePop(listdictInitPop)
        for gen in range(self.iGenNum):
            listdictPopAfCros = self.funCrossover(listdictCurrPop,
                                                  self.fCrosRate)
            listdictPopAfMuta = self.funMutation(listdictPopAfCros)
            listdictCurrPop = self.funSurvival(listdictCurrPop,
                                               listdictPopAfMuta)
        listdictFinalPop = listdictCurrPop
        return listdictFinalPop
