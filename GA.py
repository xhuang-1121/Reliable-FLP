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
        # TODO 根据目标函数定义fitness function
        fFitness = 0
        return fFitness
