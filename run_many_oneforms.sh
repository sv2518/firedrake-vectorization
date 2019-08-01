#!/bin/bash
batchsize=(1 4)  # 1: not vectorize, 4: vectorize by 4
mesh=('quad' 'tri' 'hex' 'tetra')
# mesh=('quad')
form=('helmholtz' 'mass' 'laplacian' 'elasticity' 'hyperelasticity')
# form=('helmholtz')
vs=('omp' 've')  # vectorization strategy
export PYOP2_TIME=1  # switch on timing mode
export TJ_NP=1  # number of processes
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
