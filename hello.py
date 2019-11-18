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

def do_something(x):
    v = pow(x, 2)
    return v

if __name__ == '__main__':
    a =[]
    start = timeit.default_timer()
    for i in range(1, 10):
        a.append(do_something(i))

    end = timeit.default_timer()
    print('single processing time:', str(end-start), 's')
    print(a[1:10])

	# revise to parallel
    items = [x for x in range(1, 10)]
    p = multiprocessing.Pool()
    start = timeit.default_timer()
    b = p.map(do_something, items)
    p.close()
    p.join()
    end = timeit.default_timer()
    print('multi processing time:', str(end-start), 's')
    print(b)
    print('Return values are all equal ?:', operator.eq(a, b))