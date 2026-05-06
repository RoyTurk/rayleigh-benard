"""
Time integration solver for Rayleigh-Bénard convection.

Drives the time loop with Picard iteration for the nonlinear
coupled momentum-energy system, and handles XDMF output and statistics.
"""
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx import fem
from mpi4py import MPI
import numpy as np
import os


class Solver:
    """
    Time-stepping solver for the Rayleigh-Bénard convection problem.

    Advances the solution using BDF1 (implicit Euler) with a Picard
    iteration loop to handle nonlinearity. Velocity/pressure and temperature
    are solved sequentially at each Picard iterate. Writes XDMF output and
    saves time-series statistics (Nusselt number, Picard convergence) to disk.

    Parameters
    ----------
    t0 : float
        Start time.
    Tend : float
        End time.
    num_steps : int
        Number of uniform time steps.
    problem : Problem
        Configured Problem instance containing mesh, forms, and BCs.
    max_iter : int
        Maximum Picard iterations per time step.
    tol : float
        Convergence tolerance for Picard iteration (L2 norm of increment).
    """
    def __init__(self, t0, Tend, num_steps, problem, max_iter=20, tol=1e-6):
        
        self.max_iter = max_iter
        self.tol = tol
        
        self.t0 = t0
        self.Tend = Tend
        self.num_steps = num_steps
        self.time_array, self.dt_value = np.linspace(t0, Tend, num_steps + 1, retstep=True)
        
        self.problem = problem
        self.problem.dt.value = self.dt_value
        
        # Direct LU solvers for momentum and energy subproblems
        self.problem_up = LinearProblem(
            problem.a1,
            problem.L1,
            bcs = problem.bc_up,
            u=problem.up_sol,
            petsc_options_prefix="up_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        
        self.problem_T = LinearProblem(
            problem.a2,
            problem.L2,
            bcs=problem.bc_T,
            u=problem.T_sol,
            petsc_options_prefix="T_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        
        if self.problem.mesh_comm.rank == 0:
            os.makedirs("output", exist_ok=True)
        
        self.V_out = fem.functionspace(problem.mesh, ("Lagrange", 2, (problem.mesh.geometry.dim,)))
        self.W_out = fem.functionspace(problem.mesh, ("Lagrange", 2))
        
        self.u_out = fem.Function(self.V_out, name="velocity")
        self.p_out = fem.Function(self.W_out, name="pressure")
        self.T_out = fem.Function(self.W_out, name="temperature")
        
        self.xdmf_u = XDMFFile(problem.mesh_comm, "output/velocity.xdmf", "w")
        self.xdmf_p = XDMFFile(problem.mesh_comm, "output/pressure.xdmf", "w")
        self.xdmf_T = XDMFFile(problem.mesh_comm, "output/temperature.xdmf", "w")
        
        self.xdmf_u.write_mesh(problem.mesh)
        self.xdmf_p.write_mesh(problem.mesh)
        self.xdmf_T.write_mesh(problem.mesh)
        
        self.history = {
            "t": [],
            "step": [],
            "Nu": [],
            "picard_iters": [],
            "picard_converged": [],
        }
        
    def run(self, write_every=10):
        
        for step, t in enumerate(self.time_array[1:], start=1):
            
            if self.problem.mesh_comm.rank == 0:
                print(f"\n--- Time step {step}, t = {t:.5f} ---", flush=True)

            # Initialize Picard iterates and solution from previous time step
            self.problem.up_k.x.array[:] = self.problem.up_n.x.array
            self.problem.T_k.x.array[:] = self.problem.T_n.x.array
            self.problem.up_sol.x.array[:] = self.problem.up_n.x.array
            self.problem.T_sol.x.array[:] = self.problem.T_n.x.array
            
            picard_iters, converged = self._picard_step()
            
            # Advance solution to next time step
            self.problem.up_n.x.array[:] = self.problem.up_sol.x.array
            self.problem.T_n.x.array[:] = self.problem.T_sol.x.array
            self.problem.up_n.x.scatter_forward()
            self.problem.T_n.x.scatter_forward()
            
            if step % write_every == 0:
                self._write_output(t)
            
            Nu_local = fem.assemble_scalar(self.problem.Nu_form)
            Nu = - self.problem.mesh_comm.allreduce(Nu_local, op=MPI.SUM) / self.problem.L
            
            self.history["t"].append(t)
            self.history["step"].append(step)
            self.history["Nu"].append(Nu)
            self.history["picard_iters"].append(picard_iters)
            self.history["picard_converged"].append(converged)
            
            if self.problem.mesh_comm.rank == 0:
                print(f"Nu = {Nu:.4f}", flush=True)
        
        self.xdmf_u.close()
        self.xdmf_p.close()
        self.xdmf_T.close()
        
        if self.problem.mesh_comm.rank == 0:
            np.savez(
                "output/statistics.npz",
                t=np.array(self.history["t"]),
                step=np.array(self.history["step"]),
                Nu=np.array(self.history["Nu"]),
                picard_iters=np.array(self.history["picard_iters"]),
                picard_converged=np.array(self.history["picard_converged"]),
            )
            print("Saved statistics to output/statistics.npz", flush=True)
            
    def _picard_step(self):
        """
        Run the Picard iteration for the current time step.

        Alternately solves the momentum and energy subproblems until the
        increment in both fields falls below the tolerance, or the maximum
        number of iterations is reached.

        Returns
        -------
        iters : int
            Number of Picard iterations performed.
        converged : bool
            True if the tolerance was met, False if max_iter was reached.
        """
        for k in range(self.max_iter):
            self.problem_up.solve()
            self.problem.up_sol.x.scatter_forward()
            
            self.problem_T.solve()
            self.problem.T_sol.x.scatter_forward()
            
            err_up = self.norm(self.problem.up_sol.x.array - self.problem.up_k.x.array)
            err_T = self.norm(self.problem.T_sol.x.array - self.problem.T_k.x.array)
            
            err_up = self.problem.mesh_comm.allreduce(err_up, op=MPI.SUM)
            err_T = self.problem.mesh_comm.allreduce(err_T, op=MPI.SUM)
            
            if self.problem.mesh_comm.rank == 0:
                print(f"  Picard iter {k:02d} | err_up = {err_up:.3e}, err_T = {err_T:.3e}", flush=True)
            if err_up < self.tol and err_T < self.tol:
                if self.problem.mesh_comm.rank == 0:
                    print(f"Converged in {k + 1} iterations", flush=True)
                return k + 1, True
            
            self.problem.up_k.x.array[:] = self.problem.up_sol.x.array
            self.problem.T_k.x.array[:] = self.problem.T_sol.x.array
            
        else:
            if self.problem.mesh_comm.rank == 0:
                print("Picard reached maximum number of iterations.", flush=True)
            return self.max_iter, False
    
    def _write_output(self, t):
        
        u_collapsed = self.problem.up_sol.sub(0).collapse()
        p_collapsed = self.problem.up_sol.sub(1).collapse()

        self.u_out.interpolate(fem.Expression(u_collapsed, self.V_out.element.interpolation_points))
        self.p_out.interpolate(fem.Expression(p_collapsed, self.W_out.element.interpolation_points))
        self.T_out.interpolate(fem.Expression(self.problem.T_sol, self.W_out.element.interpolation_points))

        self.u_out.x.scatter_forward()
        self.p_out.x.scatter_forward()
        self.T_out.x.scatter_forward()

        self.xdmf_u.write_function(self.u_out, t)
        self.xdmf_p.write_function(self.p_out, t)
        self.xdmf_T.write_function(self.T_out, t)
    
    @staticmethod
    def norm(x):
        return np.linalg.norm(x)