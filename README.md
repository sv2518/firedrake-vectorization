# Experimental evaluation for Slate vectorisation in Firedrake

All experiment data lies in `csv/`, and result plots are in `plots/`.
Further, data collection and evaluation scripts are included for reproducibility.

There are two sets of experiments in this repository.

The first set of experiments is divided further into two cases.
First, the performance of action(form, x) is measured and only uses PyOP2 and TSFC for the assembly.
Second, the performance of Slate addition of some action(form, x) is measured and therefore uses PyOP2, Slate and TSFC.
The data for this experiment set can be reproduced with `run_many_oneforms.sh --vectorization`. Further, the results
can be plotted with `plot.py` and `plot_slate_vect.py`. The firedrake version required is captured under the DOI ...

The second set of data is measuring Slate vectorisation for the new local matrix-free infrastructure.
In particular, the performance of a Schur complement expression in Slate actioned on a coefficient is timed.
The data can be reproduced with `run_many_oneforms.sh --matfree`. Results can be plotted with `plot_schur.py`.
The firedrake version required is captured under the DOI ...

This repository only measures performance, but the code verification is run on the Firedrake github action
of the Slate vectorisation and the local matrix-free infrastructure.
