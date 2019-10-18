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
        '''
        for i in range(len(fp_listdictPop)):
            fp_listdictPop[i]['fitness'] = self.funEvaluateInd(
                fp_listdictPop[i]['chromosome'])
        listdictPopBefSurv = fp_listdictPop
        return listdictPopBefSurv

    def funSelectParents(self, fp_listdictPop, fp_iIndIndex=None):
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
        fFitnessSum = sum(ind['fitness'] for ind in fp_listdictPop)
        for i in range(len(fp_listdictPop)):
            fProb.append(fp_listdictPop[i]['fitness'] / fFitnessSum)
        if fp_iIndIndex == None:
            aParents = np.random.choice(fp_listdictPop, size=2, p=fProb)
            listdictParents = list(aParents)
        else:
            listdictParents.append(fp_listdictPop[fp_iIndIndex])
            listdictParents.append(
                list(np.random.choice(fp_listdictPop, size=1, p=fProb)))
        return listdictParents

    def funCrossover(self, fp_listdictPop, fp_fCrosRate, fp_iHowSelPare):
        '''
        The value of formal parameter "fp_iSelPare" determines how to choose
        parents. If fp_iSelPare==1, the "fp_iIndIndex" of "funSelectParents" 
        should be set to "None" and choose the first parents selection approach.
        Otherwise choose the second approach.
        '''
        if len(fp_listdictPop) != self.iPopSize:
            print(
                "Someting wrong. The population size before crossover is abnormal."
            )

        listdictPopAfCros = []
        for i in range(len(fp_listdictPop)):
            if np.random.rand() < fp_fCrosRate:
                aOffs1 = np.zeros((self.iIndLen, ), dtype=np.int)
                aOffs2 = np.zeros((self.iIndLen, ), dtype=np.int)
                listdictParents = []
                if fp_iHowSelPare == 1:
                    listdictParents = self.funSelectParents(fp_listdictPop)
                else:
                    listdictParents = self.funSelectParents(fp_listdictPop,
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
        return listdictPopAfCros

    def funMutation(self, fp_listdictPopAfCros):
        for i in range()