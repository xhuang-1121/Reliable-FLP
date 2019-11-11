# first import the Model class from docplex.mp
from docplex.mp.model import Model

# create one model instance, with a name
m = Model(name='telephone_production')

# by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
desk = m.continuous_var(name='desk')
cell = m.continuous_var(name='cell')

# constraint #1: desk production is greater than 100
m.add_constraint(desk >= 100)

# constraint #2: cell production is greater than 100
m.add_constraint(cell >= 100)

# constraint #3: assembly time limit
ct_assembly = m.add_constraint(0.2 * desk + 0.4 * cell <= 400)

# constraint #4: paiting time limit
ct_painting = m.add_constraint(0.5 * desk + 0.4 * cell <= 490)

m.maximize(12 * desk + 20 * cell)
s = m.solve()
m.print_information()
m.print_solution()
