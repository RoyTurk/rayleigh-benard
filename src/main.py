"""
Main entry point for the Rayleigh-Bénard convection simulation.

Sets up the physical problem and numerical solver, then runs the time integration.
"""

from problem import Problem
from solver import Solver


# Physical parameters
L = 1.0     # Domain length
h = 2.0     # Domain height
Ra = 1e5    # Rayleigh number
Pr = 0.71   # Prandtl number (air)

# Numerical parameters
t0 = 0.0    # Start time
Tend = 0.5  # End time
num_steps = 1000
num_nodes = 64  # Nodes per spatial dimension

max_iter = 20   # Maximum Picard iterations per time step
tol = 1e-6      # Picard convergence tolerance


problem = Problem(L=L, h=h,
                num_nodes=num_nodes,
                Ra=Ra, Pr=Pr)
solver = Solver(t0=t0, Tend=Tend,
                num_steps=num_steps,
                problem=problem,
                max_iter=max_iter,
                tol=tol)

# Solve 
solver.run(write_every=5)