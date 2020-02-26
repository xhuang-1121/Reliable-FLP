# import pickle
# import instanceGeneration
# from instanceGeneration import Instances

# insName = '10-nodeInstances'
# f = open(insName, 'rb')

# a = []
# print("start")
# for i in range(10):
#     ins = pickle.load(f)
#     a.append(ins)
#     print("end")
# print(a[0].aiFixedCost)
# print(a[1].aiFixedCost)

import multiprocessing
import timeit
import operator
import numpy as np
import pickle
from instanceGeneration import Instances
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def do_something(x):
    v = pow(x, 2)
    return v


def Foo_np(seed=None):
    local_state = np.random.RandomState()
    print(local_state.rand())


def test():
    pool = multiprocessing.Pool()
    pool.map(Foo_np, range(5))
    pool.close()
    pool.join()





if __name__ == '__main__':
    """ fileName = "0600-node"
    iGenNum = 250
    listfAveBestIndFitnessEveryGen = []
    listiAveDiversityMetric1EveGen = []
    listiAveFitEvaNumByThisGen = []
    fig = plt.figure()
    listGenIndex = [g for g in range(iGenNum + 1)]
    ax1 = fig.add_subplot(111)
    l1, = ax1.plot(listGenIndex, listfAveBestIndFitnessEveryGen)
    # 右方Y轴
    ax2 = ax1.twinx()
    l2, = ax2.plot(listGenIndex, listiAveDiversityMetric1EveGen, 'r')
    # l3, = ax2.plot(listGenIndex, listiAveDiversityMetric2EveGen, 'purple', linestyle='--')
    for label in ax2.yaxis.get_ticklabels():
        label.set_fontsize(8)
    # 上方X轴
    ax3 = ax1.twiny()  # 与ax1共用1个y轴，在上方生成自己的x轴
    ax3.set_xlabel("# of Fitness Evaluation")
    listfFeIndex = list(np.linspace(0, iGenNum, num=10+1))
    # print("listFeIndex:", listfFeIndex)
    listFeXCoordinate = []
    for f in range(len(listfFeIndex)):
        listFeXCoordinate.append(listiAveFitEvaNumByThisGen[int(listfFeIndex[f])])
    # print("listFeXCoordinate:", listFeXCoordinate)
    ax3.plot(listGenIndex, listfAveBestIndFitnessEveryGen)
    ax3.set_xticks(listfFeIndex)
    ax3.set_xticklabels(listFeXCoordinate, rotation=10)
    for label in ax3.xaxis.get_ticklabels():
        label.set_fontsize(6)
    plt.legend(handles=[l1, l2], labels=['l1', 'l2'], loc='best')

    ax1.set_xlabel("# of Generation")
    ax1.set_ylabel("Fitness Of Best Individual (× 1e-3)")
    ax2.set_ylabel("Diversity Metric")
    plt.savefig(fileName + '_GASLSDM_Curve(m=2)-ins'+str(7)+'.svg') """


    f = open('10-nodeInstances', 'rb')
    ins = pickle.load(f)
    x = ins.a_2d_SitesCoordi[:, 0]  # 横坐标数组，大小[0, 1]
    y = ins.a_2d_SitesCoordi[:, 1]  # 纵坐标数组
    plt.scatter(x, y, alpha=0.6, s=20)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看），s控制点的大小
    f_x = ins.a_2d_SitesCoordi[:2, 0]
    f_y = ins.a_2d_SitesCoordi[:2, 1]
    plt.scatter(f_x, f_y, alpha=1, s=30, marker='*', c='r')
    # plt.plot(list(x), list(y), 'bo', alpha=0.6, ms=2)  # 另一种绘制散点图的方法, ms控制点的大小
    plt.show()

    # data1 = [11784.86154, 11780.24259, 11732.45826, 11997.97669, 11860.6101, 12383.62024, 11906.57894, 12205.63157, 11784.86154, 11902.06717]
    # data2 = [11780.24259, 11732.45826, 11732.45826, 11732.45826, 11732.45826, 11732.45826, 11732.45826, 11732.45826, 11732.45826, 11780.24259]
    # stat,p = ttest_ind(data1,data2)
    # print("stat为：%f" %stat,"p值为：%f" %p)


    # test()
    # a =[]
    # start = timeit.default_timer()
    # for i in range(1, 10):
    #     a.append(do_something(i))

    # end = timeit.default_timer()
    # print('single processing time:', str(end-start), 's')
    # print(a[1:10])

    # # revise to parallel
    # items = [x for x in range(1, 10)]
    # p = multiprocessing.Pool()
    # start = timeit.default_timer()
    # b = p.map(do_something, items)
    # p.close()
    # p.join()
    # end = timeit.default_timer()
    # print('multi processing time:', str(end-start), 's')
    # print(b)
    # print('Return values are all equal ?:', operator.eq(a, b))
