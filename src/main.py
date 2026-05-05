from problem import Problem
from solver import Solver

# Physical parameters
L = 1.0
h = 2.0
num_nodes = 64
Ra = 1e5
Pr = 0.71

# Numerical parameters
t0 = 0.0
Tend = 0.5
num_steps = 1000
max_iter = 20
tol = 1e-6

problem = Problem(L=L, h=h,
                num_nodes=num_nodes,
                Ra=Ra, Pr=Pr)
solver = Solver(t0=t0, Tend=Tend,
                num_steps=num_steps,
                problem=problem,
                max_iter=max_iter,
                tol=tol)

solver.run(write_every=5)