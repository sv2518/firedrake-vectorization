import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import itertools


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
    "peak_bw": 2*59e9,
}


def compute(df, platform):
    df["flop"] = df["add"] + df["sub"] + df["mul"] + df["div"]
    df['bw / peak_l'] = df['byte'] / df['time'] / cpu[platform]["peak_bw"]
    df['bw / peak_u'] = df['mem'] * df['cell'] * 8 / df['time'] / cpu[platform]["peak_bw"]
    df['flop / s'] = df['flop'] * df['cell'] / df['time']
    df['flop / peak'] = df['flop'] * df['cell'] / df['time'] / cpu[platform]["peak_flop"]
    df["ai"] = df["flop"] * df["cell"] / df["byte"]
    df["time / cell"] = df["time"] / df["cell"]
    df["time / dof"] = df["time"] / df["dof"]
    df = df[:6]
    return df


# forms by meshes
plt.close('all')
plt.figure(figsize=(12, 6))

forms = ['hyperelasticity']
meshes = ["quad"]
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
y = "flop / peak"
# linpack_scale = cpu[platform]['peak_flop'] / cpu[platform]['peak_flop_linpack']

_color = ("red", "blue", "goldenrod", "black")

for form_id, form in enumerate(forms):
    for mesh_id, mesh in enumerate(meshes):
        dfs = []
        filename = "_".join([platform, form, mesh, threads, "1", vec, "gcc"]) + ".csv"
        base_df = pd.read_csv("./csv/" + filename)
        base_df = compute(base_df, platform)
        for compiler in compilers:
            filename = "_".join([platform, form, mesh, threads, simd, vec, compiler]) + ".csv"
            df = pd.read_csv("./csv/" + filename)
            df = compute(df, platform)
            df["speed up"] = base_df["time"] / df["time"]
            dfs.append(df)
        
        dfs.append(base_df)
        ax1 = plt.subplot(len(forms), len(meshes), mesh_id + form_id*len(meshes) + 1)
        marker = itertools.cycle(('o', 's', '*', '^'))
        color = itertools.cycle(_color)
        linestyle = itertools.cycle(('-', '--', '-.', ':',))
        names = compilers + ["baseline"]
        plots = []
        for df, n in zip(dfs, names):
            plot, = ax1.plot(df[x], df[y], marker=next(marker), color=next(color), linestyle=next(linestyle),
                            label=n, linewidth=2, markersize=5)
            if form_id == len(forms) - 1 and mesh_id == len(meshes) - 1:
                plots.append(plot)

        ax1.set_xticks(dfs[0][x])
        ax1.set_ylim(bottom=0, top=1.0)
        ax1.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax1.set_title(form + " - " + mesh)
        # plot = ax1.hlines(cpu[platform]["peak_flop_linpack"]/cpu[platform]["peak_flop"], 1, 6, 
        #                    color="grey", linestyle=":")
        
        # ax2 = ax1.twinx()
        # ax2.set_ylim(bottom=0, top=linpack_scale)
        # ax2.set_yticks([0.25, 0.5, 0.75, 1])
        # if form_id == len(forms) - 1 and mesh_id == len(meshes) - 1:
        #     plots.append(plot)
        
        if mesh_id == 0:
            ax1.set_ylabel("FLOP/s / Peak FLOP/s")
        else:
            plt.setp(ax1.get_yticklabels(), visible=False)
        
        # if mesh_id == len(meshes) - 1:
        #     ax2.set_ylabel("FLOP/s / LINPACK FLOP/s")
        # else:
        #     plt.setp(ax2.get_yticklabels(), visible=False)
            
        if form_id == len(forms) - 1:
            ax1.set_xlabel("Polynomial degree")
        else:
            plt.setp(ax1.get_xticklabels(), visible=False)

plt.figlegend(plots, ["GCC", "baseline"], ncol=5,
              loc = "center", bbox_to_anchor=[0.5, 0.04], frameon=True)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig("plots/"+platform + "-" + vec + ".pdf", format="pdf")


# roofline
plt.close('all')
plt.figure(figsize=(16, 5))

platform = "haswell-on-pex"  # haswell or skylake
forms = ["hyperelasticity"]
meshes = ["quad"]
compiler = "gcc"
vec = "cross-element"

setting = {
    "haswell": {
        "simds": ["1", "4"],
        "proc": "16",
        "yticks": [5, 10, 20, 50, 100, 200, 300, 500],
        "ytop": 500,
        "ybottom": 3,
        "xleft": 0.1,
    },
    "skylake": {
        "simds": ["1",  "8"],
        "proc": "32",
        "yticks": [10, 20, 50, 100, 200, 500, 1000, 2000],
        "ytop": 2000,
        "ybottom": 10,
        "xleft": 0.15,
    },
    "mymac": {
        "simds": ["4"],
        "proc": "4",
        "yticks": [1, 10, 20, 50, 100, 200, 500, 1000, 2000],
        "ytop": 2000,
        "ybottom": 1,
        "xleft": 0.15,
    },
    "haswell-on-pex": {
        "simds": ["1", "4"],
        "proc": "32" if hyperthreading else "16",
        "yticks": [1, 5, 10, 20, 50, 100, 200, 300, 500, 2000],
        "ytop": 1000,
    "ybottom": 3,
    "xleft": 0.1,
    },
}

x = "ai"
y = "flop / s"

plots = []

for idx, simd in enumerate(setting[platform]["simds"]):
    ax = plt.subplot(1, 2, idx+1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks(setting[platform]["yticks"])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    rate = cpu[platform]['peak_bw'] / 1e9
    plot, = ax.plot([0.1, cpu[platform]['peak_flop']/1e9/rate, 3000],
                    [rate*0.1, cpu[platform]['peak_flop']/1e9, cpu[platform]['peak_flop']/1e9], linewidth=2)
    # plot, = ax.plot([cpu[platform]['peak_flop_linpack']/1e9/rate, 3000],
    #                 [cpu[platform]['peak_flop_linpack']/1e9, cpu[platform]['peak_flop_linpack']/1e9],
    #                linestyle=':', color='grey')
    if idx == 1:
        linpack = [plot]

    markers = itertools.cycle(('o', 's', '*', '^', 'v'))
    colors = itertools.cycle(("red", "blue", "goldenrod", "green"))
    names = []
    for form_id, form in enumerate(forms):
        marker = next(markers)
        for mesh_id, mesh in enumerate(meshes):
            color = next(colors)
            filename = "_".join([platform, form, mesh, setting[platform]["proc"], simd, vec, compiler]) + ".csv"
            df = pd.read_csv("./csv/" + filename)
            df = compute(df, platform)
            plot, = ax.plot(df[x], df[y]/1e9, label=form+" - "+mesh, markersize=7, marker=marker, color=color,
                            linestyle='None')
            names.append(form+" - "+mesh)
            if idx == 1:
                plots.append(plot)

    ax.set_ylim(bottom=setting[platform]["ybottom"], top=setting[platform]["ytop"])
    ax.set_xlim(left=setting[platform]["xleft"], right=3000)
    ax.set_title(platform.capitalize() + (" baseline" if simd == "1" else " cross-element vectorization"))
    ax.set_ylabel("GFLOPS / s")
    ax.set_xlabel("Arithmetic intensity")

plt.subplots_adjust(bottom=0.3)
lgd = plt.figlegend(plots, names, ncol=5, 
                    loc = "center", bbox_to_anchor=[0.35, 0.1], frameon=False)
# plt.figlegend(linpack, ["LINPACK"], loc = "center", bbox_to_anchor=[0.7, 0.1], frameon=False)
plt.savefig("plots/"+"roofline-" + platform + ".pdf", format="pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')


# populate table in paper
forms = ["hyperelasticity"]
meshes = ["quad"]
compiler = "gcc"
vec = "cross-element"

setting = {
    "haswell": {
        "simd":"4",
        "proc": "16",
    },
    "skylake": {
        "simd": "8",
        "proc": "32",
    },
    "mymac": {
        "simd": "4",
        "proc": "4",
    },
    "haswell-on-pex": {
        "simd":"4",
        "proc": "32" if hyperthreading else "16"
    },
}

from collections import defaultdict

result = defaultdict(dict)

for form in forms:
    for mesh in meshes:
        for platform in ["haswell-on-pex"]:
            # baseline
            filename = "_".join([platform, form, mesh, setting[platform]["proc"], "1", vec, compiler]) + ".csv"
            df = pd.read_csv("./csv/" + filename)
            df = compute(df, platform)
            
            filename = "_".join([platform, form, mesh, setting[platform]["proc"], setting[platform]["simd"], vec, compiler]) + ".csv"
            df_speed = pd.read_csv("./csv/" + filename)
            df_speed = compute(df_speed, platform)
            df["speed up " + platform] = df["time"] / df_speed["time"]

            for idx, row in df.iterrows():
                result[(form, mesh, int(row["p"]))]['ai'] = "{0:.1f}".format(row["ai"])
                result[(form, mesh, int(row["p"]))]['extend_dof'] = "{0:d}".format(int(row["extend_dof"]))
                result[(form, mesh, int(row["p"]))]['extend_quad'] = "{0:d}".format(int(row["extend_quad"]))
                result[(form, mesh, int(row["p"]))]['speed up ' + platform] = "{0:.1f}".format(row["speed up " + platform])


string = ""
for form in forms:
    for p in range(1, 7):
        line = ["", str(p)]
        for mesh in meshes:
            res = result[(form, mesh, p)]
            line.extend([res['ai'], res['extend_dof'], res['extend_quad'], res['speed up haswell-on-pex']])#, res['speed up skylake']])
        string += " & ".join(line)
        string += "\\\\\n"
    string += "\\hline\n"
print(string)




