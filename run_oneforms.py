import subprocess
import csv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', dest='prefix', default="", type=str)
parser.add_argument('--suffix', dest='suffix', default="", type=str)
parser.add_argument('--runtype', dest='runtype', default="", type=str)
args, _ = parser.parse_known_args()

prefix = args.prefix
suffix = args.suffix
runtype = args.runtype


def get_n(mesh, p, form, runtype):
    """ The number of mesh elements depends on the polynomial order and the form. """
    if mesh == "hex":
        if form == "inner_schur":
            return 16 if "highorder" in runtype else 32
        elif form == "outer_schur":
            return 16
        else:
            if p > 7:
                return 32
            else:
                return 64
    elif mesh == "tet":
        if p > 4:
            return 32
        else:
            return 64
    elif mesh == "quad":
        if p > 9:
            return 512
        elif p > 4:
            return 1024
        else:
            return 2048
    elif mesh == "tri":
        if p > 9:
            return 256
        elif p > 4:
            return 512
        elif p > 2:
            return 1024
        else:
            return 2048
    else:
        raise AssertionError()


# Setup mesh geometry and form name
try:
    mesh = os.environ["TJ_MESH"]
except:
    mesh = "quad"

try:
    form = os.environ["TJ_FORM"]
except:
    form = "helmholtz"

try:
    np = os.environ["TJ_NP"]
except:
    np = "1"

# Handle the optimisation modes
optimise = ""
matfree = ""
prec = ""
try:
    if os.environ["SV_OPTS"] == "MOP":
        optimise =  "--optimise"
    elif os.environ["SV_OPTS"] == "FOP":
        optimise = "--optimise"
        matfree = "--matfree"
    elif os.environ["SV_OPTS"] == "PFOP":
        optimise = "--optimise"
        matfree = "--matfree"
        prec = "--prec"
except:
    pass

# Setup polonomial approximation degrees
if mesh == "hex":
    if runtype == "highordermatfree":
        if form == "inner_schur":
            ps = range(1, 10)
        else:
            ps = range(1, 8)
    elif runtype == "matfree":
        ps = range(1, 6)
    else:
        ps = range(1, 7)
elif mesh == "quad":
    ps = range(1, 7)
elif mesh == "tri":
    ps = range(1, 7)
elif mesh == "tet":
    ps = range(1, 7)
else:
    raise AssertionError()

# Use the right kernel name to access the FLOP measures etc.
if runtype == "slatevectorization":
    knl_name = "slate_wrapper"
else:
    if "inner_schur" in form:
        knl_name = "slate_loopy_knl_0" if optimise and matfree else "wrap_slate_loopy_knl_0"
    elif "outer_schur" in form:
        knl_name = "slate_loopy_knl_0" if optimise and matfree else "wrap_slate_loopy_knl_0" if optimise else "wrap_slate_loopy_knl_0"
    else:
        knl_name = "form0_cell_integral_otherwise"

fs = [0]
repeat = 5
simd_width = os.environ['PYOP2_SIMD_WIDTH']
vect_strategy = os.environ['PYOP2_VECT_STRATEGY']
compiler = os.environ['MPICH_CC']
mpi_map_by = os.environ['TJ_MPI_MAP_BY']
print("form={0}, mesh={1}, simd={2}, np={3}, {4}, {5}".format(form, mesh, simd_width, np, vect_strategy, compiler))

result = [("n", "p", "f", "dof", "cell", "add", "sub", "mul", "div", "mem", "byte", "time", "ninst", "nloops", "extend_dof", "extend_quad")]
for p in ps:
    for f in fs:
        n = get_n(mesh, p, form, runtype)
        print("n={0}, p={1}, f={2}".format(n, p, f))
        print(f"opts={optimise}{matfree}{prec}")
        cmd = ["mpiexec", "-np", np, "--bind-to", "hwthread", "--map-by", mpi_map_by,
               "python", "oneform.py", "--n", str(n), "--p", str(p), "--f", str(f),
               "--form", form, "--mesh", mesh, "--repeat", str(repeat),
               optimise, matfree, prec, "--name", knl_name, "--runtype", runtype]
        cmd.append("-log_view")

        output = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode("utf-8")
        # print(output)
        output = output.split()
        
        time = float(output[output.index(f"Parloop_Cells_wrap_{knl_name}") + 3]) / repeat
        dofs = int(output[output.index("DOFS=") + 1])
        cells = int(output[output.index("CELLS=") + 1])
        adds = int(output[output.index("ADDS=") + 1])
        subs = int(output[output.index("SUBS=") + 1])
        muls = int(output[output.index("MULS=") + 1])
        divs = int(output[output.index("DIVS=") + 1])
        mems = int(output[output.index("MEMS=") + 1])
        bytes = int(output[output.index(f"wrap_{knl_name}_BYTES=") + 1])
        instructions = int(output[output.index("INSTRUCTIONS=") + 1])
        loops = int(output[output.index("LOOPS=") + 1])
        dof_loop_extent = int(output[output.index("DOF_LOOP_EXTENT=") + 1])
        quadrature_loop_extent = int(output[output.index("QUADRATURE_LOOP_EXTENT=") + 1])

        print("dofs={0}, cells={1}, adds={2}, subs={3}, muls={4}, divs={5}, mems={6}, bytes={7}, time={8}, "
              "inst={9}, loops={10}, dof_loop_extent={11}, quadrature_loop_extent={12}".format(
            dofs, cells, adds, subs, muls, divs, mems, bytes, time, instructions, loops, dof_loop_extent, quadrature_loop_extent))
        result.append((n, p, f, dofs, cells, adds, subs, muls, divs, mems, bytes, time,
                       instructions, loops, dof_loop_extent, quadrature_loop_extent))

# This dance was required because I changed the
# naming convention changed between the different experiments
# FIXME This could be avoided by avoided by rerunning everything
# and use a consistent naming convention
if runtype == "highordermatfree":
    name = "homatfslateexpr_"
    suffix += "_optimise{10_matfree{1}_prec{2}".format(bool(optimise), bool(matfree), bool(prec))
elif runtype == "matfree":
    name = "matfslateexpr_"
    suffix += "_optimise{0}_matfree{1}_prec{2}".format(bool(optimise), bool(matfree), bool(prec))
elif runtype == "slatevectorization":
    name = "slateexpr_"
    if simd_width == 1:
        vect_strategy == "cross-element"
else:
    name = ""
    if simd_width == 1:
        vect_strategy == "cross-element"
csvfile = open('csv/{0}{1}_{2}{3}_{4}_{5}_{6}{7}.csv'.format(prefix, form, name, mesh, str(np), simd_width, vect_strategy, suffix), 'w')
writer = csv.writer(csvfile)
writer.writerows(result)
csvfile.close()
