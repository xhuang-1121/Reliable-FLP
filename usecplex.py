from docplex.mp.model import Model
from docplex.cp.model import CpoModel
import instanceGeneration


class CPLEX:
    def __init__(self, fp_listCplexParameters, fp_obInstance):
        '''
        @fp_listCplexParameters: [0:iCandidateFaciNum, 1:fAlpha]
        '''
        self.iCandidateFaciNum = fp_listCplexParameters[0]
        self.fAlpha = fp_listCplexParameters[1]
        self.obInstance = fp_obInstance
        self.model = Model()
        self.cpomodel = CpoModel()

    def fun_fillModel(self):
        # creat decision variables list
        listDeciVarX = self.model.binary_var_list(self.iCandidateFaciNum, lb=0, name='X')
        listDeciVarY = self.model.binary_var_list(pow(self.iCandidateFaciNum, 3), lb=0, name='Y')
        # construct objective function
        objFunction = 0
        for i in range(self.iCandidateFaciNum):
            objFunction += self.obInstance.aiFixedCost[i] * listDeciVarX[i]
        listTranCost = []
        for i in range(self.iCandidateFaciNum):
            for j in range(self.iCandidateFaciNum):
                for r in range(2):  # 只分配两个设施
                    fTranCost = self.fAlpha * self.obInstance.aiDemands[i] * self.obInstance.af_2d_TransCost[i][j] * pow(self.obInstance.fFaciFailProb, r) * (1 - self.obInstance.fFaciFailProb)
                    listTranCost.append(fTranCost)
                    objFunction += fTranCost * listDeciVarY[pow(self.iCandidateFaciNum, 2) * i + self.iCandidateFaciNum * j + r]
        # for i in range(pow(self.iCandidateFaciNum, 2) * 2):
        #     objFunction += listTranCost[i] * listDeciVarY[i]
        self.model.minimize(objFunction)
        # add constraints
        for i in range(self.iCandidateFaciNum):
            for r in range(2):  # 只分配两个设施
                cons1 = 0
                for j in range(self.iCandidateFaciNum):
                    cons1 += listDeciVarY[i * pow(self.iCandidateFaciNum, 2) + j * self.iCandidateFaciNum + r]
                self.model.add_constraint(cons1 == 1)  # constraint 1

        for i in range(self.iCandidateFaciNum):
            for j in range(self.iCandidateFaciNum):
                cons3 = 0
                for r in range(2):  # 只分配两个设施
                    cons3 += listDeciVarY[i * pow(self.iCandidateFaciNum, 2) + j * self.iCandidateFaciNum + r]
                self.model.add_constraint(cons3 <= listDeciVarX[j])  # constraint 3

        cons4 = 0
        for j in range(self.iCandidateFaciNum):
            cons4 += listDeciVarX[j]
        self.model.add_constraint(cons4 >= 2)

    def fun_fillCpoModel(self):
        # creat decision variables list
        listDeciVarX = self.cpomodel.binary_var_list(self.iCandidateFaciNum, name='X')
        listDeciVarY = self.cpomodel.binary_var_list(pow(self.iCandidateFaciNum, 3), name='Y')
        # construct objective function
        objFunction = 0
        for i in range(self.iCandidateFaciNum):
            objFunction += self.obInstance.aiFixedCost[i] * listDeciVarX[i]
        listTranCost = []
        for i in range(self.iCandidateFaciNum):
            for j in range(self.iCandidateFaciNum):
                for r in range(2):  # 只分配两个设施
                    listTranCost.append(self.fAlpha * self.obInstance.aiDemands[i] * self.obInstance.af_2d_TransCost[i][j] * pow(self.obInstance.fFaciFailProb, r) * (1 - self.obInstance.fFaciFailProb))
        for i in range(pow(self.iCandidateFaciNum, 2) * 2):
            objFunction += listTranCost[i] * listDeciVarY[i]
        self.cpomodel.add(self.cpomodel.minimize(objFunction))
        # add constraints
        for i in range(self.iCandidateFaciNum):
            for r in range(2):  # 只分配两个设施
                cons1 = 0
                for j in range(self.iCandidateFaciNum):
                    cons1 += listDeciVarY[i * pow(self.iCandidateFaciNum, 2) + j * self.iCandidateFaciNum + r]
                self.cpomodel.add(cons1 == 1)  # constraint 1

        for i in range(self.iCandidateFaciNum):
            for j in range(self.iCandidateFaciNum):
                cons3 = 0
                for r in range(2):  # 只分配两个设施
                    cons3 += listDeciVarY[i * pow(self.iCandidateFaciNum, 2) + j * self.iCandidateFaciNum + r]
                self.cpomodel.add(cons3 <= listDeciVarX[j])  # constraint 3

        cons4 = 0
        for j in range(self.iCandidateFaciNum):
            cons4 += listDeciVarX[j]
        self.cpomodel.add(cons4 >= 2)


if __name__ == '__main__':
    '''
    @listInstPara=[0:iSitesNum, 1:iScenNum, 2:iDemandLB, 3:iDemandUB, 4:iFixedCostLB, 5:iFixedCostUP, 6:iCoordinateLB, 7:iCoordinateUB, 8:fFaciFailProb]

    @fp_listCplexParameters: [0:iCandidateFaciNum, 1:fAlpha]
    '''
    listInstPara = [30, 1, 0, 1000, 100, 1000, 0, 1, 0.05]
    # Generate instance
    obInstance = instanceGeneration.Instances(listInstPara)
    obInstance.funGenerateInstances()
    # cplex
    listCplexParameters = [30, 1]
    cplexSolver = CPLEX(listCplexParameters, obInstance)
    # cplexSolver.fun_fillModel()
    # sol = cplexSolver.model.solve()
    cplexSolver.fun_fillCpoModel()
    sol = cplexSolver.cpomodel.solve(TimeLimit=1)
    print("Solution status: " + sol.get_solve_status())
    # print(sol)
