import cplex
from cplex.exceptions import CplexError

# data common to all populateby functions
my_obj = [1.0, 2.0, 3.0, 1.0]  # 目标函数的系数 
my_ub = [40.0, cplex.infinity, cplex.infinity, 3.0]  # 各变量的上界
my_lb = [0.0, 0.0, 0.0, 2.0]  # 各变量的下界
my_ctype = "CCCI"
my_colnames = ["x1", "x2", "x3", "x4"]
my_rhs = [20.0, 30.0, 0.0]
my_rownames = ["r1", "r2", "r3"]
my_sense = "LLE"


def populatebyrow(prob):
    prob.objective.set_sense(prob.objective.sense.maximize)

    prob.variables.add(obj=my_obj, lb=my_lb, ub=my_ub, types=my_ctype,
                       names=my_colnames)

    rows = [[["x1", "x2", "x3", "x4"], [-1.0, 1.0, 1.0, 10.0]],
            [["x1", "x2", "x3"], [1.0, -3.0, 1.0]],
            [["x2", "x4"], [1.0, -3.5]]]

    prob.linear_constraints.add(lin_expr=rows, senses=my_sense,
                                rhs=my_rhs, names=my_rownames)


try:
    my_prob = cplex.Cplex()
    handle = populatebyrow(my_prob)
    my_prob.solve()

except CplexError as exc:
    print(exc)

print()
# solution.get_status() returns an integer code
print("Solution status = ", my_prob.solution.get_status(), ":", end=' ')
# the following line prints the corresponding string
print(my_prob.solution.status[my_prob.solution.get_status()])
print("Solution value  = ", my_prob.solution.get_objective_value())

numcols = my_prob.variables.get_num()
numrows = my_prob.linear_constraints.get_num()

slack = my_prob.solution.get_linear_slacks()
x = my_prob.solution.get_values()

print('x: ')
print(x)
