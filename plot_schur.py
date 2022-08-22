import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import itertools
import seaborn as sns
from math import log

plt.style.use("seaborn-darkgrid")
sns.set(font_scale=1.15)

def compute(df, platform):
    df["DOF / time [s]"] = df["dof"] / df["time"]
    return df

runs = ["matfslateexpr", "homatfslateexpr"]
number_opts = [4, 1]
opts_list = [(False, False, False), (True, False, False), (True, True, False), (True, True, True)]
names = ["not optimised", "expression order optimised", "matfree", "preconditioned matfree", "vectorised preconditioned matfree"]
values = [slice(5), slice(None)]
yranges = [[(1e4, 5e6), (5e4, 5e7)],[(1e5, 3e6), (5e5, 5e7)]]

for yrange, (value, (number_opt, runtype)) in zip(yranges, zip(values, zip(number_opts, runs))): 
    forms = ["outer_schur", "inner_schur"]
    meshes = ["hex"]
    opts = opts_list[4-number_opt:4]
    names = names[4-number_opt:5]
    platform = "haswell-on-pex"
    hyperthreading = True
    vec = "cross-element"

    if platform == "haswell":
        simd = "4"
        if hyperthreading:
            threads = "16"
        else:
            threads = "8"
    elif platform == "mymac":
        simd = "4"
        threads = "16"
    elif platform == "haswell-on-pex":
        simd = "4"
        if hyperthreading:
            threads = "32"
        else:
            threads = "16"
    else:
        simd = "8"
        threads = "32" if hyperthreading else "16"
        
    compilers = ["gcc"]
    x = "p"
    y = "DOF / time [s]"
    palette = sns.color_palette(n_colors=5)

    for form_id, form in enumerate(forms):
        plt.close('all')
        plt.figure(figsize=(10, 5))
        for mesh_id, mesh in enumerate(meshes):
            dfs = []
            filename = "_".join([platform, form+'_'+runtype, mesh, threads, "4", "cross-element", "gcc", "optimiseTrue", "matfreeTrue", "precTrue"]) + ".csv"
            base_df = pd.read_csv("./csv/" + filename)
            base_df = compute(base_df, platform)
            for compiler in compilers:
                for opt in opts:
                    filename = "_".join([platform, form+'_'+runtype, mesh, threads, "4", "", compiler, f"optimise{opt[0]}", f"matfree{opt[1]}", f"prec{opt[2]}"]) + ".csv"
                    df = pd.read_csv("./csv/" + filename)
                    df = compute(df, platform)
                    df["speed up"] = base_df["time"] / df["time"]
                    dfs.append(df)
            
            dfs.append(base_df)
            ax1 = plt.subplot(1, len(meshes), mesh_id + 0*len(meshes) + 1)
            marker = itertools.cycle(('o', 's', '*', '^', "D"))
            color = itertools.cycle((palette[0], palette[3], palette[1], palette[2], palette[4]))
            linestyle = itertools.cycle(('dotted', '--', '-.', '-', (0, (3, 1, 1, 1, 1, 1))))
            plots = []
            for df, n in zip(dfs, names):
                plot, = ax1.plot(df[x][value], df[y][value], marker=next(marker), color=next(color), linestyle=next(linestyle),
                                label=n, linewidth=2, markersize=5)
                if mesh_id == len(meshes) - 1:
                    plots.append(plot)

            ax1.set_xticks(dfs[0][x][value])
            ax1.set_ylabel(y, weight="bold", labelpad=10)
            ax1.set_xlabel("Polynomial degree\n [DOFS]", weight="bold", labelpad=10)
            ax1.set_xticklabels(list(f"{degree}\n[{dof}]" for degree, dof in zip(dfs[0][x][value], dfs[0]["dof"][value])))
            ax1.set_yticks([5**i for i in range(int(log(yrange[form_id][0])), int(log(yrange[form_id][1])))])
            ax1.set_ylim(bottom=yrange[form_id][0], top=yrange[form_id][1])

        plt.legend(plots, names, frameon=True,
                      facecolor='white', fancybox=True, loc='best')
        ax1.set_yscale('log')
        plt.grid(True, which="both", color='white')
        plt.tight_layout()
        plt.savefig(f"plots/slate/{runtype}-{form}-{platform}-{vec}.pdf", format="pdf")

