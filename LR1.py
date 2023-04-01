# -*- coding: UTF-8 -*-
import instanceGeneration
import numpy as np
import copy


class LagrangianRelaxation:
    def __init__(self, fp_listParameters, fp_obInstance):
        '''
        @fp_listParameters=[0:iMaxIterationNum, 1:fBeta, 2:fBetaMin, 3:fAlpha, 4:fToleranceEpsilon]
        '''
        self.iMaxIterNum = fp_listParameters[0]
        self.fBeta = fp_listParameters[1]
        self.fBetaMin = fp_listParameters[2]
        self.fAlpha = fp_listParameters[3]
        self.fToleranceEpsilon = fp_listParameters[4]  # Used in funTerminateCondition()
        self.obInstance = fp_obInstance

        self.iCandidateSitesNum = self.obInstance.iSitesNum
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

    def funSolveRelaxationProblem(self):
        '''
        Solve the relaxation problem and give a lower bound of the optimal value of the original problem.
        @return: aLocaSolXj, a3dAlloSolYijr, feasible, fLowerBound
        '''
        # location decision
        aLocaSolXj = np.zeros(self.iCandidateSitesNum, dtype=np.int)
        # allocation decision
        a3dAlloSolYijr = np.zeros((self.iCandidateSitesNum, self.iCandidateSitesNum, self.iCandidateSitesNum), dtype=np.int)
        # Psi, i.e., ψ
        a3dPsi = np.zeros((self.iCandidateSitesNum, self.iCandidateSitesNum, self.iCandidateSitesNum))
        for i in range(self.iCandidateSitesNum):
            for j in range(self.iCandidateSitesNum):
                for r in range(self.iCandidateSitesNum):
                    a3dPsi[i][j][r] = self.fAlpha * self.obInstance.aiDemands[i] * self.obInstance.af_2d_TransCost[i][j] * pow(self.obInstance.fFaciFailProb, r) * (1 - self.obInstance.fFaciFailProb) - self.a2dLambda[i][r]
        i = j = r = 0
        afGamma = np.zeros((self.iCandidateSitesNum,))
        count = 0
        for j in range(self.iCandidateSitesNum):
            tempA = 0
            for i in range(self.iCandidateSitesNum):
                fMinPsi = np.min(a3dPsi[i][j])
                tempA += min(0, fMinPsi)
                # tempA += fMinPsi
            afGamma[j] = self.obInstance.aiFixedCost[j] + tempA
            if afGamma[j] < 0:
                aLocaSolXj[j] = 1
            else:
                count += 1
        i = j = 0
        if count in [self.iCandidateSitesNum, self.iCandidateSitesNum - 1]:
            # np.where() return "tuple" type data. The element of the tuple is arrays.
            aSortedGamma = sorted(afGamma)  # default ascending order
            aIndexJ = np.where(afGamma < aSortedGamma[2])[0]
            aLocaSolXj[aIndexJ[0]] = 1
            aLocaSolXj[aIndexJ[1]] = 1

        # Until now we get X_j. Next we need to determine Y_{ijr}.
        self.iRealFaciNum = np.sum(aLocaSolXj == 1)
        for i in range(self.iCandidateSitesNum):
            if self.iRealFaciNum == 1:  # 修改过后这种情况不可能出现
                # np.where() return "tuple" type data. The element of the tuple is arrays.
                faciIndex = np.where(aLocaSolXj == 1)[0][0]
                if (a3dPsi[i][faciIndex][0] < 0) and (a3dPsi[i][faciIndex][0] == np.min(a3dPsi[i][faciIndex][0])):
                    a3dAlloSolYijr[i][faciIndex][0] = 1
            else:
                for j in range(self.iCandidateSitesNum):
                    for r in range(self.iRealFaciNum):
                        if (
                            aLocaSolXj[j] == 1
                            and a3dPsi[i][j][r] < 0
                            and a3dPsi[i][j][r]
                            == np.min(a3dPsi[i][j][: self.iRealFaciNum])
                        ):
                        # if (aLocaSolXj[j] == 1) and (a3dPsi[i][j][r] == np.min(a3dPsi[i][j][0:self.iRealFaciNum])):
                            a3dAlloSolYijr[i][j][r] = 1
        i = j = r = 0
        fLowerBound = sum(
            afGamma[j] * aLocaSolXj[j] for j in range(self.iCandidateSitesNum)
        )
        j = 0
        for i in range(self.iCandidateSitesNum):
            for r in range(self.iRealFaciNum):
                for j in range(self.iCandidateSitesNum):
                    if aLocaSolXj[j] == 1 and a3dAlloSolYijr[i][j][r] == 1:
                        fLowerBound += self.a2dLambda[i][r]
            # fLowerBound += np.min(self.a2dLambda[i]) # 或者用max?
            # fLowerBound += self.a2dLambda[i][r]
        # fLowerBound += np.sum(self.a2dLambda)  # sum(map(sum, self.a2dLambda))

        # Until now we get Y_{ijr}. Next we should check whether Y_{ijr} is feasible for original problem.
        feasible = self.funCheckFeasible(a3dAlloSolYijr)
        return aLocaSolXj, a3dAlloSolYijr, feasible, fLowerBound

    def funCheckFeasible(self, fp_a3dAlloSolYijr):
        for i in range(self.iCandidateSitesNum):
            for r in range(self.iRealFaciNum):
                constraint1 = sum(
                    fp_a3dAlloSolYijr[i][j][r]
                    for j in range(self.iCandidateSitesNum)
                )
                if constraint1 != 1:
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
        if iSelcSitesNum == 0:
            return 0
        for i in range(self.iCandidateSitesNum):  # i represents different customers.
            aSelcSitesTransCostForI = np.multiply(
                fp_aLocaSolXj, self.obInstance.af_2d_TransCost[i])

            aSelcSitesTransCostForI = [
                value for value in aSelcSitesTransCostForI if value != 0
            ]
            # if site i is selected, it would be missed in the above step and its trans cost is 0.
            if fp_aLocaSolXj[i] == 1:
                aSelcSitesTransCostForI = np.append(aSelcSitesTransCostForI, 0)
            if iSelcSitesNum != len(aSelcSitesTransCostForI):
                print("Wrong in funUpperBound(). Please check.")
            aSortedTransCostForI = sorted(aSelcSitesTransCostForI)  # ascending order

            # w1 += self.obInstance.aiDemands[i] * aSortedTransCostForI[0]

            # j represents the facilities that allocated to the customer i
            for j in range(len(aSortedTransCostForI)):
                p = self.obInstance.fFaciFailProb
                w2 += self.obInstance.aiDemands[i] * aSortedTransCostForI[
                    j] * pow(p, j) * (1 - p)

        return w1 + self.fAlpha * w2

    def funUpdateMultiplierLambda(self, fp_lowerBound_n):
        '''
        Compute the (n+1)th iteration's multiplier, i.e., λ_{ir}.
        @fp_lowBound_n: The value of lower bound for the n-th iteration.
        '''
        fTempA = 0
        arrayOfSumYijr = np.zeros((self.iCandidateSitesNum, self.iCandidateSitesNum))
        a2dLambda_nextIter = np.zeros((self.iCandidateSitesNum, self.iCandidateSitesNum))
        # "aFaciIndex" stores indexes of selected facilities.
        aFaciIndex = np.where(self.aLocaSolXj == 1)[0]
        for i in range(self.iCandidateSitesNum):
            for r in range(self.iRealFaciNum):
                sumYijr = sum(self.a3dAlloSolYijr[i][j][r] for j in aFaciIndex)
                arrayOfSumYijr[i][r] = sumYijr  # Stored and used for compute λ_(n+1)
                fTempA += pow((1 - sumYijr), 2)
        i = j = r = 0
        stepSize = self.fBeta * ((self.fBestUpperBound - fp_lowerBound_n)) / fTempA
        for i in range(self.iCandidateSitesNum):
            for r in range(self.iCandidateSitesNum):
                a2dLambda_nextIter[i][r] = self.a2dLambda[i][r] + stepSize * (1 - arrayOfSumYijr[i][r])
                # 以下出自https://www.cnblogs.com/Hand-Head/articles/8861153.html
                a2dLambda_nextIter[i][r] = max(a2dLambda_nextIter[i][r], 0)
        return a2dLambda_nextIter

    def funInitMultiplierLambda(self):
        '''
        Initialize Lagrangian multiplier λir
        '''
        fDisdanceBar = np.sum(self.obInstance.af_2d_TransCost) / pow(self.iCandidateSitesNum, 2)
        for i in range(self.iCandidateSitesNum):
            for r in range(self.iCandidateSitesNum):
                self.a2dLambda[i][r] = self.obInstance.aiDemands[i] * fDisdanceBar / pow(10, r+2)

    def funMeetTerminationCondition(self, fp_lowerBound_n, fp_n):
        '''
        @fp_lowerBound_n: The value of lower bound for the n-th iteration.
        @return: "True" represents the process should be terminated.
        '''
        if self.fBestUpperBound > fp_lowerBound_n and ((self.fBestUpperBound - fp_lowerBound_n) / fp_lowerBound_n) < self.fToleranceEpsilon:
            print("1111111111111111111")
            return True
        elif fp_n > self.iMaxIterNum:
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
        while not meetTerminationCondition:
            aLocaSolXj, self.a3dAlloSolYijr, self.feasible, fLowerBound = self.funSolveRelaxationProblem()
            fUpperBound = self.funUpperBound(aLocaSolXj)
            if fLowerBound > fUpperBound:
                print("Whether LB < UP? : ", n, fLowerBound < fUpperBound)
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
            self.a2dLambda = self.funUpdateMultiplierLambda(fLowerBound)
            meetTerminationCondition = self.funMeetTerminationCondition(fLowerBound, n)
            n += 1
        print("n: ", n)
        print("UB update number: ", UBupdateNum)
        print("LB update number: ", LBupdateNum)
        print("Xj: ", self.aLocaSolXj)
        print("Upper bound: ", self.fBestUpperBound)
        print("Lower bound: ", self.fBestLowerBoundZLambda)
        print("Gap: ", (self.fBestUpperBound - self.fBestLowerBoundZLambda) / self.fBestUpperBound)
        return self.fBestUpperBound, self.fBestLowerBoundZLambda


if __name__ == '__main__':
    '''
    listParameters=[0:iMaxIterationNum, 1:fBeta, 2:fBetaMin, 3:fAlpha, 4:fToleranceEpsilon]

    listInstPara=[0:iSitesNum, 1:iScenNum, 2:iDemandLB, 3:iDemandUB, 4:iFixedCostLB, 5:iFixedCostUP, 6:iCoordinateLB, 7:iCoordinateUB, 8:fFaciFailProb]
    '''
    listParameters = [600, 2.0, 1e-8, 1.0, 0.0001]
    listInstPara = [10, 1, 0, 1000, 500, 1500, 0, 1, 0.05]
    # Generate instance
    obInstance = instanceGeneration.Instances(listInstPara)
    obInstance.funGenerateInstances()
    # Lagrangian relaxation
    meetTerminationCondition = False
    fLowerBound = 0
    fUpperBound = 0
    n = 0  # Iteration number
    nonImproveIterNum = 0
    LR = LagrangianRelaxation(listParameters, obInstance)
    LR.funInitMultiplierLambda()
    UBupdateNum = 0
    LBupdateNum = 0
    while not meetTerminationCondition:
        aLocaSolXj, LR.a3dAlloSolYijr, LR.feasible, fLowerBound = LR.funSolveRelaxationProblem()
        fUpperBound = LR.funUpperBound(aLocaSolXj)
        if fLowerBound > fUpperBound:
            print("Whether LB < UP? : ", n, fLowerBound < fUpperBound)
        if fUpperBound < LR.fBestUpperBound:
            LR.fBestUpperBound = fUpperBound
            LR.aLocaSolXj = copy.deepcopy(aLocaSolXj)
            UBupdateNum += 1
        # if fLowerBound > LR.fBestLowerBoundZLambda and fLowerBound < LR.fBestUpperBound:
        # if fLowerBound > 0 and fLowerBound < LR.fBestUpperBound and fLowerBound > LR.fBestLowerBoundZLambda:
            LR.fBestLowerBoundZLambda = fLowerBound
            LBupdateNum += 1
        else:
            nonImproveIterNum += 1
            if nonImproveIterNum == 30:
                LR.fBeta /= 2
                nonImproveIterNum = 0
        if LR.feasible is True:
            print("Feasible solution found.")
            break
        LR.a2dLambda = LR.funUpdateMultiplierLambda(fLowerBound)
        meetTerminationCondition = LR.funMeetTerminationCondition(fLowerBound, n)
        n += 1
    print("n: ", n)
    print("UB update number: ", UBupdateNum)
    print("LB update number: ", LBupdateNum)
    print("Xj: ", LR.aLocaSolXj)
    print("Upper bound: ", LR.fBestUpperBound)
    print("Lower bound: ", LR.fBestLowerBoundZLambda)
    print("Gap: ", (LR.fBestUpperBound - LR.fBestLowerBoundZLambda) / LR.fBestLowerBoundZLambda)
    print()
