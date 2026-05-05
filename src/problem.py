from dolfinx import fem
from dolfinx.io import gmsh as gmshio
from petsc4py import PETSc
from mpi4py import MPI
import gmsh
import numpy as np
from basix.ufl import element, mixed_element
from ufl import (TrialFunction,
                TestFunction,
                split,
                dot, inner,
                grad, div,
                nabla_grad,
                as_vector,
                dx,
                ds)

class Problem:
    
    def __init__(self,
                L=1.0, h=1.0, 
                num_nodes=51, 
                Ra=1e3, Pr=0.71
                ):

        self.mesh_comm = MPI.COMM_WORLD
        self.model_rank = 0
        
        self.L = L
        self.h = h
        
        self.Ra = Ra
        self.Pr = Pr

        # Geometric dimension
        gdim = 2
        
        self.fluid_tag = 1
        self.bottom_tag, self.top_tag, self.side_tag = 2, 3, 4
                
        gmsh.initialize()
        
        if self.mesh_comm.rank == 0:
            
            gmsh.model.occ.addRectangle(0, 0, 0, L, h, tag=1)
            gmsh.model.occ.synchronize()
            
            surfaces = gmsh.model.getEntities(dim=gdim)
            boundaries = gmsh.model.getBoundary(surfaces, oriented=False)
            
            self._classify_boundaries(surfaces, boundaries)
            
            for curve in gmsh.model.occ.getEntities(1):
                gmsh.model.mesh.setTransfiniteCurve(curve[1], num_nodes)
            for surf in gmsh.model.occ.getEntities(2):
                gmsh.model.mesh.setTransfiniteSurface(surf[1])
            
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
            
            gmsh.model.mesh.generate(2)
            # gmsh.fltk.run()


            
        mesh_data = gmshio.model_to_mesh(gmsh.model, self.mesh_comm, self.model_rank, gdim=gdim)
        self.mesh = mesh_data.mesh
        self.ft = mesh_data.facet_tags
        self.ft.name = "Facet markers"
        
        gmsh.finalize()
        
        V_cg2 = element("Lagrange", self.mesh.basix_cell(), 2, shape=(self.mesh.geometry.dim,))
        Q_cg1 = element("Lagrange", self.mesh.basix_cell(), 1)
        W_cg2 = element("Lagrange", self.mesh.basix_cell(), 2)
        
        self.Z = fem.functionspace(self.mesh, mixed_element([V_cg2, Q_cg1]))
        self.W = fem.functionspace(self.mesh, W_cg2)
        
        fdim = self.mesh.topology.dim - 1
        
        bottom_dofs = fem.locate_dofs_topological(self.W, fdim, self.ft.find(self.bottom_tag))
        bcT_bottom = fem.dirichletbc(PETSc.ScalarType(1.0), bottom_dofs, self.W)
        
        top_dofs = fem.locate_dofs_topological(self.W, fdim, self.ft.find(self.top_tag))
        bcT_top = fem.dirichletbc(PETSc.ScalarType(0.0), top_dofs, self.W)
        
        V_sub, _ = self.Z.sub(0).collapse()
        u_noslip = fem.Function(V_sub)
        u_noslip.x.array[:] = 0.0
        
        wall_facets = np.concatenate([
            self.ft.find(self.bottom_tag),
            self.ft.find(self.top_tag),
            self.ft.find(self.side_tag)
        ])
        
        wall_dofs = fem.locate_dofs_topological((self.Z.sub(0), V_sub), fdim, wall_facets)
        bcu_walls = fem.dirichletbc(u_noslip, wall_dofs, self.Z.sub(0))
        
        Q_sub, _ = self.Z.sub(1).collapse()
        p_corner = fem.Function(Q_sub)
        p_corner.x.array[:] = 0.0
        
        bottom_left_dof = fem.locate_dofs_geometrical((self.Z.sub(1), Q_sub), self.bottom_left)
        bcp_corner = fem.dirichletbc(p_corner, bottom_left_dof, self.Z.sub(1))
        
        self.bc_up = [bcu_walls, bcp_corner]
        self.bc_T = [bcT_bottom, bcT_top]
        
        up = TrialFunction(self.Z)
        vq = TestFunction(self.Z)
        
        u, p = split(up)
        v, q = split(vq)
        
        T = TrialFunction(self.W)
        Theta = TestFunction(self.W)
        
        self.up_n = fem.Function(self.Z)
        u_n, _ = split(self.up_n)

        self.up_k = fem.Function(self.Z)
        u_k, _ = split(self.up_k)

        self.up_sol = fem.Function(self.Z)
        u_sol, _ = split(self.up_sol)

        self.T_n = fem.Function(self.W)
        self.T_k = fem.Function(self.W)
        self.T_sol = fem.Function(self.W)
        
        Ra_const = fem.Constant(self.mesh, PETSc.ScalarType(self.Ra))
        Pr_const = fem.Constant(self.mesh, PETSc.ScalarType(self.Pr))
        self.dt = fem.Constant(self.mesh, PETSc.ScalarType(0.0))
        
        y = as_vector((0.0, 1.0))

        # Momentum bilinear form
        a1 = (1 / (Pr_const * self.dt)) * dot(u, v) * dx
        a1 += (1 / Pr_const) * dot(dot(u_k, nabla_grad(u)), v) * dx
        a1 += inner(grad(u), grad(v)) * dx
        a1 -= p * div(v) * dx
        a1 += q * div(u) * dx

        # Momentum linear form
        L1 = (1 / (Pr_const * self.dt)) * dot(u_n, v) * dx
        L1 += Ra_const * self.T_k * dot(v, y) * dx

        self.a1 = fem.form(a1)
        self.L1 = fem.form(L1)

        # Energy bilinear form
        a2 = (1 / self.dt) * T * Theta * dx
        a2 += dot(u_sol, nabla_grad(T)) * Theta * dx
        a2 += inner(grad(T), grad(Theta)) * dx

        # Energy linear form
        L2 = (1 / self.dt) * self.T_n * Theta * dx

        self.a2 = fem.form(a2)
        self.L2 = fem.form(L2)
        
        self.Nu_form = fem.form(
            dot(grad(self.T_n), as_vector([0.0, 1.0]))
            * ds(self.bottom_tag, domain=self.mesh, subdomain_data=self.ft)
        )
        
        self.apply_initial_conditions()
        
    def _classify_boundaries(self, surfaces, boundaries):
        
        bottom_edges, top_edges, side_edges = [], [], []

        for boundary in boundaries:
            center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
            
            if np.allclose(center_of_mass, [self.L / 2, 0, 0]):
                bottom_edges.append(boundary[1])
            elif np.allclose(center_of_mass, [self.L / 2, self.h, 0]):
                top_edges.append(boundary[1])
            else:
                side_edges.append(boundary[1])
        
        gmsh.model.addPhysicalGroup(surfaces[0][0], [surfaces[0][1]], self.fluid_tag)
        gmsh.model.setPhysicalName(surfaces[0][0], self.fluid_tag, "Fluid")

        gmsh.model.addPhysicalGroup(1, bottom_edges, self.bottom_tag)
        gmsh.model.setPhysicalName(1, self.bottom_tag, "Bottom wall")

        gmsh.model.addPhysicalGroup(1, top_edges, self.top_tag)
        gmsh.model.setPhysicalName(1, self.top_tag, "Top wall")

        gmsh.model.addPhysicalGroup(1, side_edges, self.side_tag)
        gmsh.model.setPhysicalName(1, self.side_tag, "Side walls")
    
    def bottom_left(self, x):
        return np.logical_and.reduce(
            (np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))
        )
    
    def apply_initial_conditions(self):
        
        def u0(x):
            u0 = np.zeros((2, x.shape[1]))
            u0[0] = - 64 * x[0]**2 * (x[0] - 1)**2 * x[1] * (x[1] - 1) * (2 * x[1] - 1)
            u0[1] = 64 * x[0] * (x[0] - 1) * (2 * x[0] - 1) * x[1]**2 * (x[1] - 1)**2
            return u0
        
        def T0(x):
            eps = 0.1
            rng = np.random.default_rng(seed=67)
            return 1.0 - x[1] / self.h + eps * rng.uniform(-1.0, 1.0, x.shape[1])
        
        self.up_n.sub(0).interpolate(u0)
        self.up_n.x.scatter_forward()
        
        self.T_n.interpolate(T0)
        self.T_n.x.scatter_forward()