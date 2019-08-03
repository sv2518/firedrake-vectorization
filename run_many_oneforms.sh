#!/bin/bash
arch='haswell'
# arch="skylake"
compiler=('gcc' 'clang' 'icc')
# compiler=('gcc')
batchsize=(1 4)  # 1: not vectorize, 4: vectorize by 4
mesh=('quad' 'tri' 'hex' 'tet')
# mesh=('quad')
form=('helmholtz' 'mass' 'laplacian' 'elasticity' 'hyperelasticity')
# form=('helmholtz')
vs=('omp' 've')  # vectorization strategy
# vs=('ve')
export PYOP2_TIME=1  # switch on timing mode
export TJ_NP=16  # number of processes

for v in ${vs[@]}
do
    for m in ${mesh[@]}
    do
        for f in ${form[@]}
        do
            for bs in ${batchsize[@]}
            do
                for comp in ${compiler[@]}
                do
                    export TJ_FORM=$f
                    export TJ_MESH=$m
                    export PYOP2_SIMD_WIDTH=$bs
                    export PYOP2_VECT_STRATEGY=$v
                    export OMPI_CC=$comp
                    export PYOP2_CFLAGS="-march=native"
                    if [ $comp == "icc" ]
                    then
                        if [ $arch == "haswell" ]
                        then
                            export PYOP2_CFLAGS="-xcore-avx2"
                        else
                            export PYOP2_CFLAGS="-xcore-avx512 -qopt-zmm-usage=high"
                        fi
                    fi
                    python run_oneforms.py --prefix "$arch"_ --suffix "_$comp"
                    firedrake-clean
                done
            done
        done
    done
done
