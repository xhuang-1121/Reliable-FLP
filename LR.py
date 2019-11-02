import instanceGeneration
import numpy as np


class LagrangianRelaxation:
    def __init__(self, fp_listParameters, fp_obInstance):
        '''
        @fp_listParameters=[0:iMaxIterationNum, 1:fInitalStepSize, 2:fBeta, 3: a2dLagMultiplier, 4:fAlpha]
        '''
        self.iMaxIterNum = fp_listParameters[0]
        self.fInitStepSize = fp_listParameters[1]
        self.fBeta = fp_listParameters[2]
        self.a2dLambda = fp_listParameters[3]  # lambda: Lagrangian multiplier
        self.fAlpha = fp_listParameters[4]
        self.obInstance = fp_obInstance

        self.iCandidateSitesNum = self.obInstance.iSitesNum
        # location decision
        self.aSelcSolXj = np.zeros(self.iCandidateSitesNum, dtype=np.int)
        # allocation decision
        self.a3dAlloSolYijr = np.zeros((self.iCandidateSitesNum, self.iCandidateSitesNum, self.iCandidateSitesNum), dtype=np.int)
        self.fLowerBoundZLambda = 0

    def funSolveRelaxationProblem(self):
        '''
        Solve the relaxation problem and give a lower bound of the optimal value of the original problem.
        '''
        # location decision
        aSelcSolXj = np.zeros(self.iCandidateSitesNum, dtype=np.int)
        # allocation decision
        a3dAlloSolYijr = np.zeros((self.iCandidateSitesNum, self.iCandidateSitesNum, self.iCandidateSitesNum), dtype=np.int)
        # Psi, i.e., ψ
        a3dPsi = np.zeros((self.iCandidateSitesNum, self.iCandidateSitesNum, self.iCandidateSitesNum))
        for i in range(self.iCandidateSitesNum):
            for j in range(self.iCandidateSitesNum):
                for r in range(self.iCandidateSitesNum):
                    a3dPsi[i][j][r] = self.fAlpha * self.obInstance.aiDemands[i] * self.obInstance.af_2d_TransCost * pow(self.obInstance.fFaciFailProb, r) * (1 - self.obInstance.fFaciFailProb) - self.a2dLambda[i][r]

        afGamma = np.zeros((self.iCandidateSitesNum,))
        for j in range(self.iCandidateSitesNum):
            tempA = 0
            count = 0
            for i in range(self.iCandidateSitesNum):
                fMinPsi = np.min(a3dPsi[i][j])
                tempA += min(0, fMinPsi)
            afGamma[j] = self.obInstance.aiFixedCost + tempA
            if afGamma[j] < 0:
                aSelcSolXj[j] = 1
            else:
                count += 1

        if count == self.iCandidateSitesNum:
            # np.where() return "tuple" type data. The element of the tuple is arrays.
            indexJ = np.where(afGamma == np.min(afGamma))[0][0]
            aSelcSolXj[indexJ] = 1

        for j in range(self.iCandidateSitesNum):
            self.fLowerBoundZLambda += afGamma[j] * aSelcSolXj[j]
        self.fLowerBoundZLambda += sum(map(sum, self.a2dLambda))

        # Until now we get X_j and the lower bound. Next we need to determine Y_{ijr}.
        iRealFaciNum = np.sum(aSelcSolXj == 1)
        for i in range(self.iCandidateSitesNum):
            if iRealFaciNum == 1:
                # np.where() return "tuple" type data. The element of the tuple is arrays.
                faciIndex = np.where(aSelcSolXj == 1)[0][0]
                if (a3dPsi[i][faciIndex][0] < 0) and (a3dPsi[i][faciIndex][0] == np.min(a3dPsi[i][faciIndex][0])):
                    a3dAlloSolYijr[i][faciIndex][0] = 1
            else:
                for j in range(self.iCandidateSitesNum):
                    for r in range(iRealFaciNum):
                        if (aSelcSolXj[j] == 1) and (a3dPsi[i][j][r] < 0) and (a3dPsi[i][j][r] == np.min(a3dPsi[i][j][0:iRealFaciNum])):
                            a3dAlloSolYijr[i][j][r] = 1
        # Until now we get Y_{ijr}. Next we should check whether Y_{ijr} is feasible for original problem.
        # TODO 检查是否是可行解
        feasible = self.funCheckFeasible(iRealFaciNum, a3dAlloSolYijr)
        return aSelcSolXj, a3dAlloSolYijr, feasible

    def funCheckFeasible(self, fp_iRealFaciNum, fp_a3dAlloSolYijr):
        for i in range(self.iCandidateSitesNum):
            for r in range(fp_iRealFaciNum):
                constraint1 = 0
                for j in range(self.iCandidateSitesNum):
                    constraint1 += fp_a3dAlloSolYijr[i][j][r]
                if constraint1 != 1:
                    return False
        return True
