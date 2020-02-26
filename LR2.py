# -*- coding: UTF-8 -*-
import instanceGeneration
import GA
import usecplex
from docplex.mp.model import Model
import numpy as np
import copy
import matplotlib.pyplot as plt
import sys


class LagrangianRelaxation:
    def __init__(self, fp_listParameters, fp_obInstance):
        '''
        @fp_listParameters=[0:iMaxIterationNum, 1:fBeta, 2:fBetaMin, 3:fAlpha, 4:fToleranceEpsilon, 5:boolAllo2Faci, 6:boolCallCplexOrNot, 7:iHowToAlloYijr: 1or2or3]
        '''
        self.model = Model()  # 调用cplex mp model

        self.iMaxIterNum = fp_listParameters[0]
        self.fBeta = fp_listParameters[1]
        self.fBetaMin = fp_listParameters[2]
        self.fAlpha = fp_listParameters[3]
        self.fToleranceEpsilon = fp_listParameters[4]  # Used in funTerminateCondition()
        self.boolAllo2Faci = fp_listParameters[5]
        self.boolCallCplexOrNot = fp_listParameters[6]
        self.iHowToAlloYijr = fp_listParameters[7]  # 该参数只有不调用cplex的时候才起作用
        self.obInstance = fp_obInstance

        self.iCandidateSitesNum = self.obInstance.iSitesNum
        self.iCandidateFaciNum = self.obInstance.iSitesNum
        # creat decision variables list
        self.listDeciVarX = self.model.binary_var_list(self.iCandidateFaciNum, lb=0, name='X')
        self.listDeciVarY = self.model.binary_var_list(pow(self.iCandidateFaciNum, 3), lb=0, name='Y')
        # lambda: Lagrangian multiplier
        self.a2dLambda = np.zeros((self.iCandidateSitesNum, self.iCandidateSitesNum))
        # location decision
        self.aLocaSolXj = np.zeros(self.iCandidateSitesNum, dtype=np.int)
        # allocation decision
        self.a3dAlloSolYijr = np.zeros((self.iCandidateSitesNum, self.iCandidateSitesNum, self.iCandidateSitesNum), dtype=np.int)
        self.fBestLowerBoundZLambda = 0
        self.fBestUpperBound = float('inf')
        self.iRealFaciNum = 0
        self.feasible = False

    def funInitUpperBound_greedy(self):
        aLocaSolXj = np.ones(self.iCandidateSitesNum, dtype=np.int)
        self.iRealFaciNum = np.sum(aLocaSolXj == 1)
        fUB = self.funUpperBound(aLocaSolXj)
        if fUB < self.fBestUpperBound:
            self.fBestUpperBound = fUB
        boolContinue = True
        while boolContinue:
            iUpdateNum = 0
            for i in range(self.iCandidateSitesNum):
                if aLocaSolXj[i] == 1:
                    aLocaSolXj[i] = 0
                    self.iRealFaciNum = np.sum(aLocaSolXj == 1)
                    fUB = self.funUpperBound(aLocaSolXj)
                    if fUB < self.fBestUpperBound:
                        self.fBestUpperBound = fUB
                        self.aLocaSolXj = copy.deepcopy(aLocaSolXj)
                        iUpdateNum += 1
                    aLocaSolXj[i] = 1
            if iUpdateNum == 0:
                boolContinue = False
        return

    def funSolveRelaxationProblem(self):
        '''
        Solve the relaxation problem and give a lower bound of the optimal value of the original problem.
        @return: aLocaSolXj, a3dAlloSolYijr, feasible, fLowerBound
        '''
        # location decision
        aLocaSolXj = np.zeros(self.iCandidateSitesNum, dtype=np.int)
        # allocation decision
        a3dAlloSolYijr = np.zeros((self.iCandidateSitesNum, self.iCandidateSitesNum, self.iCandidateSitesNum), dtype=np.int)
        # relaxation problem's object value, lower bound
        fLowerBound = 0
        # Psi, i.e., ψ
        a3dPsi = np.zeros((self.iCandidateSitesNum, self.iCandidateSitesNum, self.iCandidateSitesNum))
        # Phi, i.e., φ
        aPhi = np.zeros((self.iCandidateSitesNum,))
        # compute Psi
        if self.boolAllo2Faci is True:
            iAlloFaciNum = 2
        else:
            iAlloFaciNum = self.iCandidateSitesNum
        for i in range(self.iCandidateSitesNum):
            for j in range(self.iCandidateSitesNum):
                for r in range(iAlloFaciNum):
                    a3dPsi[i][j][r] = self.fAlpha * self.obInstance.aiDemands[i] * self.obInstance.af_2d_TransCost[i][j] * pow(self.obInstance.fFaciFailProb, r) * (1 - self.obInstance.fFaciFailProb) + self.a2dLambda[i][j]
        i = j = r = 0
        # compute Phi
        for j in range(self.iCandidateSitesNum):
            fSumLambda = 0
            for i in range(self.iCandidateSitesNum):
                fSumLambda += self.a2dLambda[i][j]
            aPhi[j] = self.obInstance.aiFixedCost[j] - fSumLambda
        i = j = 0
        # determine Xj
        count = 0
        for j in range(self.iCandidateSitesNum):
            if aPhi[j] < 0:
                aLocaSolXj[j] = 1
            else:
                count += 1
        if count == self.iCandidateSitesNum or count == (self.iCandidateSitesNum - 1):
            aSortedPhi = sorted(aPhi)  # default increasing order
            aIndexJ = np.where(aPhi <= aSortedPhi[2])[0]
            aLocaSolXj[aIndexJ[0]] = 1
            aLocaSolXj[aIndexJ[1]] = 1
        j = 0
        # Until now we get X_j. Next we need to determine Y_{ijr}.
        self.iRealFaciNum = np.sum(aLocaSolXj == 1)
        if self.boolAllo2Faci is True:
            iAlloFaciNum = 2
        else:
            iAlloFaciNum = self.iRealFaciNum
        
        # 怎样分配Yijr
        if self.iHowToAlloYijr == 1:
            # 1-下面这段，j可以作为同一i的不同r
            for i in range(self.iCandidateSitesNum):
                aPsiIJ0 = np.array([a3dPsi[i][0][0]])
                for j in range(1, self.iCandidateSitesNum):
                    aPsiIJ0 = np.append(aPsiIJ0, a3dPsi[i][j][0])
                iIndexOfMinPsiFaciJ = np.where(aPsiIJ0 == min(aPsiIJ0))[0][0]
                for r in range(iAlloFaciNum):
                    a3dAlloSolYijr[i][iIndexOfMinPsiFaciJ][r] = 1
                    fLowerBound += a3dPsi[i][iIndexOfMinPsiFaciJ][r]
        elif self.iHowToAlloYijr == 2:
            # 2-下面这段，j只能作为同一i的某一个r
            for i in range(self.iCandidateSitesNum):
                aPsiIJ0 = np.array([a3dPsi[i][0][0]])
                for j in range(1, self.iCandidateSitesNum):
                    aPsiIJ0 = np.append(aPsiIJ0, a3dPsi[i][j][0])
                aSortedPsiIJ0 = sorted(aPsiIJ0)  # default increasing order, 对于同一i的某一r下，所有j的Psi的大小顺序是一样的，这里用r==0时来排序
                for r in range(iAlloFaciNum):
                    iIndexOfFaciJ = np.where(aPsiIJ0 == aSortedPsiIJ0[r])[0][0]
                    a3dAlloSolYijr[i][iIndexOfFaciJ][r] = 1
                    # Compute lower bound
                    fLowerBound += a3dPsi[i][iIndexOfFaciJ][r]
        else:
            # 3-下面分配Yijr, 参考了Snyder2005的文章
            for j in range(self.iCandidateSitesNum):
                if aLocaSolXj[j] == 1:
                    for i in range(self.iCandidateSitesNum):
                        for r in range(iAlloFaciNum):
                            if a3dPsi[i][j][r] < 0:
                                aPsiIJr = []
                                for r2 in range(iAlloFaciNum):
                                    aPsiIJr.append(a3dPsi[i][j][r2])
                                aSortedPsiIJr = sorted(aPsiIJr)
                                if a3dPsi[i][j][r] == aSortedPsiIJr[0]:
                                    a3dAlloSolYijr[i][j][r] = 1
                                    # Compute lower bound
                                    fLowerBound += a3dPsi[i][j][r]

        # Compute lower bound
        fLowerBound += np.dot(aPhi, aLocaSolXj)

        # Until now we get Y_{ijr}. Next we should check whether Y_{ijr} is feasible for original problem.
        # feasible = self.funCheckFeasible(aLocaSolXj, a3dAlloSolYijr)
        feasible = False
        return aLocaSolXj, a3dAlloSolYijr, feasible, fLowerBound

    def funSolveRelaxationProblem_useCplex(self):
        '''
        Solve the relaxation problem and give a lower bound of the optimal value of the original problem.
        @return: aLocaSolXj, a3dAlloSolYijr, feasible, fLowerBound
        '''
        # location decision
        aLocaSolXj = np.zeros(self.iCandidateSitesNum, dtype=np.int)
        # allocation decision
        a3dAlloSolYijr = np.zeros((self.iCandidateSitesNum, self.iCandidateSitesNum, self.iCandidateSitesNum), dtype=np.int)
        # relaxation problem's object value, lower bound
        fLowerBound = 0
        # Psi, i.e., ψ
        a3dPsi = np.zeros((self.iCandidateSitesNum, self.iCandidateSitesNum, self.iCandidateSitesNum))
        # Phi, i.e., φ
        aPhi = np.zeros((self.iCandidateSitesNum,))
        # compute Psi
        if self.boolAllo2Faci is True:
            iAlloFaciNum = 2
        else:
            iAlloFaciNum = self.iCandidateSitesNum
        for i in range(self.iCandidateSitesNum):
            for j in range(self.iCandidateSitesNum):
                for r in range(iAlloFaciNum):
                    a3dPsi[i][j][r] = self.fAlpha * self.obInstance.aiDemands[i] * self.obInstance.af_2d_TransCost[i][j] * pow(self.obInstance.fFaciFailProb, r) * (1 - self.obInstance.fFaciFailProb) + self.a2dLambda[i][j]
        i = j = r = 0
        # compute Phi
        for j in range(self.iCandidateSitesNum):
            fSumLambda = 0
            for i in range(self.iCandidateSitesNum):
                fSumLambda += self.a2dLambda[i][j]
            aPhi[j] = self.obInstance.aiFixedCost[j] - fSumLambda
        i = j = 0

        # use cplex to get lower bound
        self.fun_fillMpModel_Allo2Faci(aPhi, a3dPsi)
        sol = self.model.solve()
        for i in range(self.iCandidateFaciNum):
            aLocaSolXj[i] = sol.get_value('X_'+str(i))
            # if sol.get_value('X_'+str(i)) == 1:
            #     print('X_'+str(i)+" =", 1)
        self.iRealFaciNum = np.sum(aLocaSolXj == 1)
        iAlloNum = 0
        for i in range(self.iCandidateFaciNum):
            for j in range(self.iCandidateFaciNum):
                for r in range(self.iCandidateFaciNum):
                    a3dAlloSolYijr[i][j][r] = sol.get_value('Y_'+str(pow(self.iCandidateFaciNum, 2) * i + self.iCandidateFaciNum * j + r))
                    # if sol.get_value('Y_'+str(pow(self.iCandidateFaciNum, 2) * i + self.iCandidateFaciNum * j + r)) == 1:
                    #     iAlloNum += 1
                    #     print('Y_'+str(pow(self.iCandidateFaciNum, 2) * i + self.iCandidateFaciNum * j + r)+" =", 1)

        fLowerBound = sol.get_objective_value()
        # feasible = self.funCheckFeasible(aLocaSolXj, a3dAlloSolYijr)
        feasible = False
        return aLocaSolXj, a3dAlloSolYijr, feasible, fLowerBound

    def fun_fillMpModel_Allo2Faci(self, aPhi, a3dPsi):
        # # creat decision variables list, 这个放在了__init__函数中，否则总是警告变量名重复
        # listDeciVarX = self.model.binary_var_list(self.iCandidateFaciNum, lb=0, name='X')
        # listDeciVarY = self.model.binary_var_list(pow(self.iCandidateFaciNum, 3), lb=0, name='Y')

        # construct objective function
        objFunction = 0
        for j in range(self.iCandidateFaciNum):
            objFunction += aPhi[j] * self.listDeciVarX[j]
        for i in range(self.iCandidateFaciNum):
            for j in range(self.iCandidateFaciNum):
                for r in range(2):  # 只分配两个设施
                    objFunction += a3dPsi[i][j][r] * self.listDeciVarY[pow(self.iCandidateFaciNum, 2) * i + self.iCandidateFaciNum * j + r]

        self.model.minimize(objFunction)
        # add constraints
        for i in range(self.iCandidateFaciNum):
            for r in range(2):  # 只分配两个设施
                cons1 = 0
                for j in range(self.iCandidateFaciNum):
                    cons1 += self.listDeciVarY[i * pow(self.iCandidateFaciNum, 2) + j * self.iCandidateFaciNum + r]
                self.model.add_constraint(cons1 == 1)  # constraint 1

        cons4 = 0
        cons4 = sum(self.listDeciVarX)
        # for j in range(self.iCandidateFaciNum):
        #     cons4 += listDeciVarX[j]
        self.model.add_constraint(cons4 >= 2)

    def funCheckFeasible(self, fp_aLocaSolXj, fp_a3dAlloSolYijr):
        for i in range(self.iCandidateSitesNum):
            for j in range(self.iCandidateSitesNum):
                if sum(fp_a3dAlloSolYijr[i][j]) > fp_aLocaSolXj[j]:
                    return False
        return True

    def funUpperBound(self, fp_aLocaSolXj):
        '''
        @fp_aLocaSolXj: facility location decision
        Compute an upper bound of the original problem.
        '''
        fUpperBound = 0
        w1 = 0
        w2 = 0
        if fp_aLocaSolXj.size != self.iCandidateSitesNum or self.obInstance.aiFixedCost.size != self.iCandidateSitesNum:
            print("Wrong. Please make sure that the size of variable \"fp_aLocaSolXj\" and \"self.obInstance.aiFixedCost\" correct.")
        w1 += np.dot(fp_aLocaSolXj, self.obInstance.aiFixedCost)
        iSelcSitesNum = np.sum(fp_aLocaSolXj)
        if self.iRealFaciNum != iSelcSitesNum:
            print("Wrong. Something wrong in funUpperBound().")
        if iSelcSitesNum == 0:
            print("Wrong. Something wrong in upperBound().")
        for i in range(self.iCandidateSitesNum):  # i represents different customers.
            aSelcSitesTransCostForI = np.multiply(
                fp_aLocaSolXj, self.obInstance.af_2d_TransCost[i])

            aSelcSitesTransCostForI = [
                value for (index, value) in enumerate(aSelcSitesTransCostForI)
                if value != 0
            ]
            # if site i is selected, it would be missed in the above step and its trans cost is 0.
            if fp_aLocaSolXj[i] == 1:
                aSelcSitesTransCostForI = np.append(aSelcSitesTransCostForI, 0)
            if iSelcSitesNum != len(aSelcSitesTransCostForI):
                print("Wrong in funUpperBound(). Please check.")
            aSortedTransCostForI = sorted(aSelcSitesTransCostForI)  # ascending order

            # j represents the facilities that allocated to the customer i
            if self.boolAllo2Faci is True:
                iAlloFaciNum = 2
            else:
                iAlloFaciNum = iSelcSitesNum
            # for j in range(iSelcSitesNum):  # 把所有Xj=1的点都分给i
            # for j in range(2):  # 每个i只有两个级别的供应点
            for j in range(iAlloFaciNum):
                p = self.obInstance.fFaciFailProb
                w2 += self.obInstance.aiDemands[i] * aSortedTransCostForI[
                    j] * pow(p, j) * (1 - p)

        fUpperBound = w1 + self.fAlpha * w2
        return fUpperBound

    def funUpdateMultiplierLambda(self, fp_aLocaSolXj, fp_lowerBound_n):
        '''
        Compute the (n+1)th iteration's multiplier, i.e., λ_{ir}.
        @fp_lowBound_n: The value of lower bound for the n-th iteration.
        '''
        denominatorOfStepSize = 0
        arrayOfSumYijr = np.zeros((self.iCandidateSitesNum, self.iCandidateSitesNum))
        a2dLambda_nextIter = np.zeros((self.iCandidateSitesNum, self.iCandidateSitesNum))
        # "aFaciIndex" stores indexes of selected facilities.
        for i in range(self.iCandidateSitesNum):
            for j in range(self.iCandidateSitesNum):
                sumYijr = 0
                for r in range(self.iRealFaciNum):
                    sumYijr += self.a3dAlloSolYijr[i][j][r]
                arrayOfSumYijr[i][j] = sumYijr  # Stored and used for compute λ_(n+1)
                denominatorOfStepSize += pow((sumYijr - fp_aLocaSolXj[j]), 2)
        i = j = r = 0
        stepSize = self.fBeta * ((self.fBestUpperBound - fp_lowerBound_n)) / denominatorOfStepSize
        for i in range(self.iCandidateSitesNum):
            for j in range(self.iCandidateSitesNum):
                a2dLambda_nextIter[i][j] = self.a2dLambda[i][j] + stepSize * (arrayOfSumYijr[i][j] - fp_aLocaSolXj[j])
                # 以下出自https://www.cnblogs.com/Hand-Head/articles/8861153.html
                if a2dLambda_nextIter[i][j] < 0:
                    a2dLambda_nextIter[i][j] = 0
        # print((a2dLambda_nextIter == self.a2dLambda).all())
        return a2dLambda_nextIter

    def funInitMultiplierLambda(self):
        '''
        Initialize Lagrangian multiplier λir
        '''
        for i in range(self.iCandidateSitesNum):
            for j in range(self.iCandidateSitesNum):
                self.a2dLambda[i][j] = self.obInstance.aiDemands[i] / self.iCandidateSitesNum

    def funMeetTerminationCondition(self, fp_lowerBound_n, fp_n):
        '''
        @fp_lowerBound_n: The value of lower bound for the n-th iteration.
        @return: "True" represents the process should be terminated.
        '''
        # if fp_lowerBound_n < self.fBestUpperBound and ((self.fBestUpperBound - fp_lowerBound_n) / self.fBestUpperBound) < self.fToleranceEpsilon:
        #     print("1111111111111111111")
        #     return True
        if fp_n > self.iMaxIterNum:
            print("222222222222222222222")
            return True
        elif self.fBeta < self.fBetaMin:
            print("333333333333333333")
            return True
        else:
            return False

    def funLR_main(self):
        meetTerminationCondition = False
        fLowerBound = 0
        fUpperBound = 0
        n = 0  # Iteration number
        nonImproveIterNum = 0
        UBupdateNum = 0
        LBupdateNum = 0
        self.funInitUpperBound_greedy()
        while meetTerminationCondition is False:
            if self.boolCallCplexOrNot is False:  # 不调用cplex
                aLocaSolXj, self.a3dAlloSolYijr, self.feasible, fLowerBound = self.funSolveRelaxationProblem()
            else:  # 调用cplex
                aLocaSolXj, self.a3dAlloSolYijr, self.feasible, fLowerBound = self.funSolveRelaxationProblem_useCplex()
            fUpperBound = self.funUpperBound(aLocaSolXj)
            if fLowerBound > fUpperBound:
                print("Whether LB < UP? : ", n, fLowerBound < fUpperBound)
            # if fUpperBound < self.fBestUpperBound:
            #     self.fBestUpperBound = fUpperBound
            #     self.aLocaSolXj = aLocaSolXj
            #     UBupdateNum += 1
            #     if fUpperBound < self.fBestLowerBoundZLambda or (fUpperBound > self.fBestLowerBoundZLambda and fLowerBound > self.fBestLowerBoundZLambda):
            #         self.fBestLowerBoundZLambda = fLowerBound
            #         LBupdateNum += 1
            # elif fLowerBound < self.fBestUpperBound and fLowerBound > self.fBestLowerBoundZLambda:
            #     self.fBestLowerBoundZLambda = fLowerBound
            #     LBupdateNum += 1
            if fUpperBound < self.fBestUpperBound:
                self.fBestUpperBound = fUpperBound
                self.aLocaSolXj = copy.deepcopy(aLocaSolXj)
                UBupdateNum += 1
            if fLowerBound > self.fBestLowerBoundZLambda:
                self.fBestLowerBoundZLambda = fLowerBound
                LBupdateNum += 1
            else:
                nonImproveIterNum += 1
                if (nonImproveIterNum % 30) == 0:
                    self.fBeta /= 2
                    nonImproveIterNum = 0
            # if self.feasible is True:
            #     print("Feasible solution found.")
            #     break
            self.a2dLambda = self.funUpdateMultiplierLambda(aLocaSolXj, fLowerBound)
            meetTerminationCondition = self.funMeetTerminationCondition(fLowerBound, n)
            n += 1
            print("n: ", n)
        print("UB update number: ", UBupdateNum)
        print("LB update number: ", LBupdateNum)
        # print("Xj: ", self.aLocaSolXj)
        print("Upper bound: ", self.fBestUpperBound)
        print("Lower bound: ", self.fBestLowerBoundZLambda)
        print("Gap: ", (self.fBestUpperBound - self.fBestLowerBoundZLambda) / self.fBestUpperBound)
        return self.fBestUpperBound, self.fBestLowerBoundZLambda


if __name__ == '__main__':
    '''
    @listLRParameters=[0:iMaxIterationNum, 1:fBeta, 2:fBetaMin, 3:fAlpha, 4:fToleranceEpsilon, 5:boolAllo2Faci, 6:boolCallCplexOrNot, 7:iHowToAlloYijr: 1or2or3]

    @listInstPara=[0:iSitesNum, 1:iScenNum, 2:iDemandLB, 3:iDemandUB, 4:iFixedCostLB, 5:iFixedCostUP, 6:iCoordinateLB, 7:iCoordinateUB, 8:fFaciFailProb]

    @fp_listCplexParameters: [0:iCandidateFaciNum, 1:fAlpha]

    @listGAParameters = [0:iGenNum, 1:iPopSize, 2:iIndLen, 3:fCrosRate, 4:fMutRate, 5:fAlpha]

    @sys.argv = [0:xx.py, 1:cplex/notcplex, 2:1/2/3, 3:iMaxIterationNum]
    '''
    if sys.argv[1] == 'cplex':
        boolCallCplexOrNot = True
    else:
        boolCallCplexOrNot = False

    if sys.argv[2] == '1':
        iHowToAlloYijr = 1
    elif sys.argv[2] == '2':
        iHowToAlloYijr = 2
    else:
        iHowToAlloYijr = 3

    iMaxIterationNum = int(sys.argv[3])

    iCandidateFaciNum = 10
    listLRParameters = [iMaxIterationNum, 2.0, 1e-8, 1.0, 0.001, True, boolCallCplexOrNot, iHowToAlloYijr]
    listInstPara = [iCandidateFaciNum, 1, 0, 1000, 500, 1500, 0, 1, 0.05]
    # Generate instance
    obInstance = instanceGeneration.Instances(listInstPara)
    obInstance.funGenerateInstances()
    # Lagrangian relaxation
    LR = LagrangianRelaxation(listLRParameters, obInstance)
    LR.funInitMultiplierLambda()
    LR.funLR_main()
    print("-------------------------------------------------------------")
    # cplex-mp module
    listCplexParameters = [iCandidateFaciNum, 1]
    cplexSolver = usecplex.CPLEX(listCplexParameters, obInstance)
    cplexSolver.fun_fillMpModel()
    sol = cplexSolver.model.solve()
    cplexSolver.model.print_information()
    optimialValue = sol.get_objective_value()
    print("Objective value: ", optimialValue)
    print(sol.solve_details)  # 获取解的详细信息，如时间，gap值等
    for i in range(cplexSolver.iCandidateFaciNum):
        if sol.get_value('X_'+str(i)) == 1:
            print('X_'+str(i)+" =", 1)
    print("-------------------------------------------------------------")
    # # genetic algorithm
    # iGenNum = 10
    # iPopSize = 10
    # boolAllo2Faci = True
    # listGAParameters = [iGenNum, iPopSize, iCandidateFaciNum, 0.9, 0.1, 1, boolAllo2Faci]
    # for i in range(5):
    #     geneticAlgo = GA.GA(listGAParameters, obInstance)
    #     finalPop, listGenNum, listfBestIndFitness = geneticAlgo.funGA_main()
    #     plt.figure()
    #     plt.plot(listGenNum, listfBestIndFitness)
    #     plt.xlabel("# of Generation")
    #     plt.ylabel("Fitness Of Best Individual")
    #     plt.show()
    #     # print(finalPop[0]['chromosome'])
    #     print(1/finalPop[0]['fitness'])
    #     print((finalPop[0]['objectValue'] - optimialValue)/finalPop[0]['objectValue'])
    #     print("-------------------------------------------------------------")

    # cplex-cp module
    # cplexSolver.fun_fillCpoModel()
    # cpsol = cplexSolver.cpomodel.solve(RelativeOptimalityTolerance=0.001, TimeLimit=10)
    # print("Solution status: " + cpsol.get_solve_status())
    # for i in range(cplexSolver.iCandidateFaciNum):
    #     if cpsol.get_all_var_solutions()[i].get_value() == 1:
    #         print(cpsol.get_all_var_solutions()[i].get_name() + " =", cpsol.get_all_var_solutions()[i].get_value())  # 打印出Xj==1的决策变量
