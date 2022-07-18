import subprocess
import csv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', dest='prefix', default="", type=str)
parser.add_argument('--suffix', dest='suffix', default="", type=str)
args, _ = parser.parse_known_args()

prefix = args.prefix
suffix = args.suffix


def get_n(mesh, p):
    if mesh == "hex":
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

try:
    if os.environ["SV_OPT"] == "NOP":
        opts = (False, False)
    elif os.environ["SV_OPT"] == "MOP":
        opts = (True, False)
    elif os.environ["SV_OPT"] == "FOP":
        opts = (True, True)
except:
    opts = (False, False)


if mesh == "hex":
    ps = range(1, 7)
elif mesh == "quad":
    ps = range(1, 7)
elif mesh == "tri":
    ps = range(1, 7)
elif mesh == "tet":
    ps = range(1, 7)
else:
    raise AssertionError()

if "schur" in form:
    knl_name = "wrap_slate_loopy_knl_7" if opts[1] and opts[0] else "wrap_wrap_slate_loopy_knl_2" if opts[0] else "wrap_wrap_slate_loopy_knl_0"
else:
    knl_name = "wrap_form0_cell_integral_otherwise"

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
        n = get_n(mesh, p)
        print("n={0}, p={1}, f={2}".format(n, p, f))
        print("opts="+str(opts))
        cmd = ["mpiexec", "-np", np, "--bind-to", "hwthread", "--map-by", mpi_map_by,
               "python", "oneform.py", "--n", str(n), "--p", str(p), "--f", str(f),
               "--form", form, "--mesh", mesh, "--repeat", str(repeat), "--optimise", opts[0], "--matfree", opts[1]]
        cmd.append("-log_view")

        output = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode("utf-8").split()
        
        time = float(output[output.index(f"Parloop_Cells_{knl_name}") + 3]) / repeat
        dofs = int(output[output.index("DOFS=") + 1])
        cells = int(output[output.index("CELLS=") + 1])
        adds = int(output[output.index("ADDS=") + 1])
        subs = int(output[output.index("SUBS=") + 1])
        muls = int(output[output.index("MULS=") + 1])
        divs = int(output[output.index("DIVS=") + 1])
        mems = int(output[output.index("MEMS=") + 1])
        bytes = int(output[output.index(f"{knl_name}_BYTES=") + 1])
        instructions = int(output[output.index("INSTRUCTIONS=") + 1])
        loops = int(output[output.index("LOOPS=") + 1])
        dof_loop_extent = int(output[output.index("DOF_LOOP_EXTENT=") + 1])
        quadrature_loop_extent = int(output[output.index("QUADRATURE_LOOP_EXTENT=") + 1])

        print("dofs={0}, cells={1}, adds={2}, subs={3}, muls={4}, divs={5}, mems={6}, bytes={7}, time={8}, "
              "inst={9}, loops={10}, dof_loop_extent={11}, quadrature_loop_extent={12}".format(
            dofs, cells, adds, subs, muls, divs, mems, bytes, time, instructions, loops, dof_loop_extent, quadrature_loop_extent))
        result.append((n, p, f, dofs, cells, adds, subs, muls, divs, mems, bytes, time,
                       instructions, loops, dof_loop_extent, quadrature_loop_extent))

csvfile = open('csv/{0}{1}_slateexpr_{2}_{3}_{4}_{5}{6}.csv'.format(prefix, form, mesh, str(np), simd_width, vect_strategy, suffix), 'w')
writer = csv.writer(csvfile)
writer.writerows(result)
csvfile.close()
