import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import itertools
import seaborn as sns

plt.style.use("seaborn-darkgrid")
sns.set(font_scale=1.15)
runtype = "_homatfslateexpr"
throughput = True

# ### Plot
# Peak performance
cpu = {}
cpu["haswell"] = {
    "peak_flop": 2.6e9 * 8 * 4 * 2 * 2,  # clock x cores x simd x fma x port
    "peak_flop_linpack": 262.5e9,  # 1 core = 37.4
    "peak_bw": 38.48e9,
}
cpu["skylake"] = {
    "peak_flop": 2.1e9 * 16 * 8 * 2 * 2,  # clock x cores x simd x fma x port
    "peak_flop_linpack": 678.8e9, # 1 core = 62.8
    "peak_bw": 36.6e9,
}
cpu["haswell-on-pex"] = {
    "peak_flop": 2.6e9 * (2*8) * 4 * 2 * 2,  # clock x cores x simd x fma x port
    "peak_bw": 59e9,
}


def compute(df, platform):
    df["flop"] = df["add"] + df["sub"] + df["mul"] + df["div"]
    df['bw / peak_l'] = df['byte'] / df['time'] / cpu[platform]["peak_bw"]
    df['bw / peak_u'] = df['mem'] * df['cell'] * 8 / df['time'] / cpu[platform]["peak_bw"]
    df['flop / s'] = df['flop'] * df['cell'] / df['time']
    df['flop / peak'] = df['flop'] * df['cell'] / df['time'] / cpu[platform]["peak_flop"]
    df["ai"] = df["flop"] * df["cell"] / df["byte"]
    df["time / cell"] = df["time"] / df["cell"]
    df["DOF / time [s]"] = df["dof"] / df["time"]
    return df

if throughput:
    forms = ["outer_schur", "inner_schur"]
    meshes = ["hex"]
    opts = [(True, True, True)]
    names = ["preconditioned matfree", "vectorised preconditioned matfree"]
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
            filename = "_".join([platform, form+runtype, mesh, threads, "4", "cross-element", "gcc", "optimiseTrue", "matfreeTrue", "precTrue"]) + ".csv"
            base_df = pd.read_csv("./csv/" + filename)
            base_df = compute(base_df, platform)
            for compiler in compilers:
                for opt in opts:
                    filename = "_".join([platform, form+runtype, mesh, threads, "4", "", compiler, f"optimise{opt[0]}", f"matfree{opt[1]}", f"prec{opt[2]}"]) + ".csv"
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
                plot, = ax1.plot(df[x], df[y], marker=next(marker), color=next(color), linestyle=next(linestyle),
                                label=n, linewidth=2, markersize=5)
                if form_id == len(forms) - 1 and mesh_id == len(meshes) - 1:
                    plots.append(plot)

            ax1.set_xticks(dfs[0][x])
            ax1.set_ylabel(y, weight="bold", labelpad=10)
            plt.setp(ax1.get_yticklabels(), visible=False)
            ax1.set_xlabel("Polynomial degree\n [DOFS]", weight="bold", labelpad=10)
            ax1.set_xticklabels(list(f"{degree}\n[{dof}]" for degree, dof in zip(dfs[0][x], dfs[0]["dof"])))

        plt.figlegend(plots, names, ncol=5,
                      loc = "center", bbox_to_anchor=[0.5, 0.04], frameon=True,
                      facecolor='white', fancybox=True)
        ax1.set_yscale('log')
        plt.grid(True, which="both", color='white')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)
        plt.savefig(f"plots/slate/ho-matf-tsslac-{form}-{platform}-{vec}.pdf", format="pdf")

