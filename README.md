# Experimental evaluation scripts for vectorisation in Firedrake

The experimental framwork is adapted from TJ Sun and his publication

  Sun, T., Mitchell, L., Kulkarni, K., Klöckner, A., Ham, D. A., & Kelly, P. H. (2020).
  A study of vectorization for matrix-free finite element methods.
  The International Journal of High Performance Computing Applications, 34(6), 629-644.
  
The experiments are extended to Slate operations on actions and actions on Slate operators, in particular Schur complements.

The performance testing framework as published on the run-on-pex branch is hardware specific.
The file run_many_oneforms.py must be altered to allow for another architecture.

All verification is performed on the github actions of https://github.com/OP2/PyOP2/tree/vectorisation-sprint and https://github.com/firedrakeproject/firedrake/pull/2365.

