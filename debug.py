from firedrake import *
from pyop2.mpi import COMM_WORLD
mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
form = inner(u, v)*dx

A = Tensor(form)
M = assemble(A + -A)
