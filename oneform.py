from firedrake import *
from forms import *
from tsfc import compile_form
from firedrake.slate.slac.compiler import compile_expression
from firedrake.slate.slate import TensorBase
import loopy as lp
import numpy as np
import argparse
from functools import reduce
import operator
from islpy import dim_type

parser = argparse.ArgumentParser()
parser.add_argument('--form', dest='form', default="helmholtz", type=str)
parser.add_argument('--p', dest='p', default=1, type=int)
parser.add_argument('--n', dest='n', default=32, type=int)
parser.add_argument('--f', dest='f', default=0, type=int)
parser.add_argument('--repeat', dest='repeat', default=1, type=int)
parser.add_argument('--mesh', dest='m', default="tri", type=str, choices=["quad", "tet", "hex", "tri"])
parser.add_argument('--print', default=False, action="store_true")
parser.add_argument('--optimise', dest='optimise', default=False, action="store_true")
parser.add_argument('--matfree', dest='matfree', default=False, action="store_true")
parser.add_argument('--name', dest='knl_name', default='slate_wrapper', type=str)
args, _ = parser.parse_known_args()

n = args.n
p = args.p
f = args.f
repeat = args.repeat
m = args.m
form_str = args.form
optimise = args.optimise
matfree = args.matfree
knl_name = args.knl_name

if m == "quad":
    mesh = IntervalMesh(n, n)
    mesh = ExtrudedMesh(mesh, n, layer_height=1.0)
    # mesh = SquareMesh(n, n, L=n, quadrilateral=True)
elif m == "tet":
    mesh = CubeMesh(n, n, n, L=n)
elif m == "hex":
    mesh = SquareMesh(n, n, L=1, quadrilateral=True)
    mesh = ExtrudedMesh(mesh, n)
else:
    assert m == "tri"
    mesh = SquareMesh(n, n, L=n)

if form_str in ["mass", "helmholtz"]:
    V = FunctionSpace(mesh, "CG", p)
elif form_str in ["laplacian", "elasticity", "hyperelasticity", "holzapfel"]:
    V = VectorFunctionSpace(mesh, "CG", p)

elif "inner_schur" in form_str:
    V = FunctionSpace(mesh, "DG", p-1)
elif "outer_schur" in form_str:
    V = FunctionSpace(mesh, "DGT", p-1)
else:
    raise AssertionError()

x = Function(V)

xs = SpatialCoordinate(mesh)
if "schur" in form_str:
    x.project(xs[0]*(1-xs[0])*xs[1]*(1-xs[1])*xs[2]*(1-xs[2]), use_slate_for_inverse=False)
else:
    if V.ufl_element().value_size() > 1:
        x.interpolate(as_vector(xs))
    else:
        x.interpolate(reduce(operator.add, xs))


form = eval(form_str)(p-1, p-1, mesh, f)
if isinstance(form, TensorBase):
    form_compiler_parameters={"slate_compiler": {"optimise": optimise, "replace_mul": matfree}}
else:
    form_compiler_parameters={}

y_form = action(form, x)
y = Function(V)
for i in range(repeat):
   assemble(y_form, tensor=y, form_compiler_parameters=form_compiler_parameters)
   y.dat.data

if args.print:
    import pickle
    pickle.dump(y.vector()[:], open("test.obj", "wb"))
    print(y.vector()[:])
    exit(0)

cells = mesh.comm.allreduce(mesh.cell_set.size)
dofs = mesh.comm.allreduce(V.dof_count)
rank = mesh.comm.Get_rank()

if rank == 0:
    if mesh.layers:
        cells = cells * (mesh.layers - 1)
    print("CELLS= {0}".format(cells))
    print("DOFS= {0}".format(dofs))

    if isinstance(y_form, TensorBase):
        tunit = compile_expression(y_form, coffee=False)[0].kinfo.kernel.code
        knl_name, = tuple(filter(lambda name: name.startswith("slate_loopy_knl"), tunit.callables_table.keys()))
    else:
        tunit = compile_form(y_form, coffee=False)[0].kinfo.kernel.code

    prog = tunit.with_entrypoints(knl_name)
    knl = prog.default_entrypoint
    warnings = list(knl.silenced_warnings)
    warnings.extend(["insn_count_subgroups_upper_bound", "no_lid_found"])
    knl = knl.copy(silenced_warnings=warnings)
    prog = prog.with_kernel(knl)
    op_map = lp.get_op_map(prog, subgroup_size=1)
    mem_map = lp.get_mem_access_map(prog, subgroup_size=1)

    for op in ['add', 'sub', 'mul', 'div']:
        print("{0}S= {1}".format(op.upper(), op_map.filter_by(name=[op], dtype=[np.float64]).eval_and_sum({})))
    print("MEMS= {0}".format(mem_map.filter_by(mtype=['global'], dtype=[np.float64]).eval_and_sum({})))
    print("INSTRUCTIONS= {0:d}".format(len(knl.instructions)))
    print("LOOPS= {0:d}".format(len(knl.all_inames())))
    for domain in knl.domains:
        if domain.get_dim_name(dim_type.set, 0)[0] == "j":
            print("DOF_LOOP_EXTENT= {0:d}".format(int(domain.dim_max_val(0).to_str()) + 1))
            break
    else:
        print("DOF_LOOP_EXTENT= 1")
    for domain in knl.domains:
        if domain.get_dim_name(dim_type.set, 0)[0:2] == "ip":
            print("QUADRATURE_LOOP_EXTENT= {0:d}".format(int(domain.dim_max_val(0).to_str()) + 1))
            break
    else:
        print("QUADRATURE_LOOP_EXTENT= 1")
