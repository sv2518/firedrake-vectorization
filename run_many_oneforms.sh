#!/bin/bash
batchsize=(1 8)
mesh=('quad' 'tri' 'hex' 'tetra')
form=('helmholtz' 'mass' 'laplacian' 'elasticity' 'hyperelasticity')
vs=('omp' 've')
for v in ${vs[@]}
do
    for m in ${mesh[@]}
    do
        for f in ${form[@]}
        do
            for bs in ${batchsize[@]}
            do
                export TJ_FORM=$f
                export TJ_MESH=$m
                export PYOP2_SIMD_WIDTH=$bs
                export PYOP2_VECT_STRATEGY=$v
                python run_oneforms.py --prefix skylake_ --suffix _icc
                firedrake-clean
            done
        done
    done
done
