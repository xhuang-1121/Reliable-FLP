# -*- coding: UTF-8 -*-
import numpy as np
import pickle


class Instances:
    def __init__(self, fp_listParameters):
        '''
        @parameters: 0:iSitesNum, 1:iScenNum, 2:iDemandLB, 3:iDemandUB, 4:iFixedCostLB, 5:iFixedCostUP, 6:iCoordinateLB, 7:iCoordinateUB, 8:fFaciFailProb
        '''
        self.iSitesNum = fp_listParameters[0]
        self.iScenNum = fp_listParameters[1]
        self.iDemandLB = fp_listParameters[2]
        self.iDemandUB = fp_listParameters[3]
        self.iFixedCostLB = fp_listParameters[4]
        self.iFixedCostUB = fp_listParameters[5]
        self.iCoordinateLB = fp_listParameters[6]
        self.iCoordinateUB = fp_listParameters[7]
        self.fFaciFailProb = fp_listParameters[8]

        self.a_2d_SitesCoordi = np.zeros((self.iSitesNum, 2))
        self.aiDemands = np.zeros(self.iSitesNum, dtype=np.int)
        self.aiFixedCost = np.zeros(self.iSitesNum, dtype=np.int)
        self.af_2d_TransCost = np.zeros((self.iSitesNum, self.iSitesNum))

    def funGenerateInstances(self):
        # generate the x and y coordinates of candidate sites
        self.a_2d_SitesCoordi = self.iCoordinateLB + (
            self.iCoordinateUB - self.iCoordinateLB) * np.random.rand(
                self.iSitesNum, 2)
        self.aiDemands = np.random.randint(self.iDemandLB,
                                           self.iDemandUB,
                                           size=self.iSitesNum)
        self.aiFixedCost = np.random.randint(self.iFixedCostLB,
                                             self.iFixedCostUB,
                                             size=self.iSitesNum)
        for i in range(self.iSitesNum):
            for j in range(i + 1, self.iSitesNum):
                temCost = np.linalg.norm(self.a_2d_SitesCoordi[i] -
                                         self.a_2d_SitesCoordi[j])
                self.af_2d_TransCost[i][j] = self.af_2d_TransCost[j][
                    i] = temCost


if __name__ == '__main__':
    '''
    Test the code.
    listPara:
    0:iSitesNum, 1:iScenNum, 2:iDemandLB, 3:iDemandUB, 4:iFixedCostLB, 5:iFixedCostUP, 6:iCoordinateLB, 7:iCoordinateUB, 8:fFaciFailProb
    '''
    iInsNum = 8
    iSitesNum = 600
    listPara = [iSitesNum, 1, 0, 1000, 500, 1500, 0, 1, 0.05]
    f = open('600-nodeInstances', 'wb')
    for i in range(iInsNum):
        generateInstances = Instances(listPara)
        generateInstances.funGenerateInstances()
        pickle.dump(generateInstances, f)
    f.close()
    # f = open('30-nodeInstances', 'rb')
    # ins1 = pickle.load(f)
    # ins2 = pickle.load(f)
    # print(ins1.aiFixedCost)
    # print(ins2.aiFixedCost)
    # print("trans cost: \n", generateInstances.af_2d_TransCost)
    # print("fixed cost: \n", generateInstances.aiFixedCost)
    # print("coordinate: \n", generateInstances.a_2d_SitesCoordi)
    # print("demands: \n", generateInstances.aiDemands)
