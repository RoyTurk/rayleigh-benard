# Rayleigh-Bénard Convection Solver

A finite element solver for 2D Rayleigh-Bénard convection built from scratch using [FEniCSx](https://fenicsproject.org/). 

---

Rayleigh-Bénard convection is a fundamental fluid dynamics problem where a layer of fluid is heated from below and cooled from above. When the temperature difference is large enough, the fluid becomes unstable and develops convection cells. Hot fluid rises, cold fluid sinks.

The key quantity of interest is the Nusselt number. It represents the ratio of total heat transfer to purely conductive heat transfer. $Nu = 1$ means no convection, $Nu > 1$ means convection is enhancing heat transfer.

---

## Governing Equations

The problem is described by the incompressible Navier-Stokes equations under the Boussinesq approximation, where density variations are only accounted for in the buoyancy term. 

**Momentum**

$$\frac{1}{Pr}\left(\partial_t \mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u}\right) = -\nabla p + \nabla^2\mathbf{u} + Ra\, T\hat{y}$$

**Continuity**

$$\nabla \cdot \mathbf{u} = 0$$

**Energy**

$$\partial_t T + (\mathbf{u}\cdot\nabla)T = \nabla^2 T$$

The two dimensionless parameters are:
- **Rayleigh number**: ratio of buoyancy to diffusive forces
- **Prandtl number**: ratio of viscous to thermal diffusion

---

## Numerical Method

- Spatial discretization: Finite element method using Taylor-Hood elements (P2/P1) for velocity pressure and P2 for temperature
- Time discretization: Backward Euler 
- Nonlinear solver: Picard iteration
- Linear solver: LU direct factorization via PETSc
- Mesh: Structured quadrilateral mesh generated with Gmsh

---

## Validation

Validated against the benchmark of Sanderse & Trias (2023) for $Ra = 10^3, 10^4, 10^5$ with $Pr=0.71$ on a unit suqare domain.

---

## Project Structure

```
src/
├── main.py       # Entry point — define parameters and run
├── problem.py    # Mesh, function spaces, BCs, and variational forms
└── solver.py     # Time loop, Picard iteration, output, and diagnostics
output/
├── velocity.xdmf
├── pressure.xdmf
├── temperature.xdmf
└── statistics.npz
```

---

## Dependencies

- FEniCSx (DOLFINx)
- Gmsh
- PETSc / petsc4py
- mpi4py
- NumPy

---

## References

- Sanderse, B. & Trias, F.X. (2023). *Energy-consistent discretization of viscous dissipation with application to natural convection flow*. arXiv:2307.10874
