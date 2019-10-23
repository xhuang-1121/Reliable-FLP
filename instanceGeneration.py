import numpy as np


class Instances:
    def __init__(self, fp_listParameters):
        '''
        @parameters: 0-iSitesNum, 1-iScenNum, 2-iDemandLB, 3-iDemandUB, 4-iFixedCostLB, 5-iFixedCostUP, 6-iCoordinateLB, 7-iCoordinateUB
        '''
        self.iSitesNum = fp_listParameters[0]
        self.iScenNum = fp_listParameters[1]
        self.iDemandLB = fp_listParameters[2]
        self.iDemandUB = fp_listParameters[3]
        self.iFixedCostLB = fp_listParameters[4]
        self.iFixedCostUB = fp_listParameters[5]
        self.iCoordinateLB = fp_listParameters[6]
        self.iCoordinateUB = fp_listParameters[7]

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
