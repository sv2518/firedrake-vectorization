#!/bin/bash
arch="mymac"
hyperthreading=0
compiler=('clang')
if [ $arch == "haswell" ]
then
    batchsize=(1 4)  # 1: not vectorize, 4: vectorize by 4
    if [ $hyperthreading == 1 ]
    then
        export TJ_NP=16  # number of processes
        export TJ_MPI_MAP_BY="hwthread"
    else
        export TJ_NP=8
        export TJ_MPI_MAP_BY="core"
    fi
elif [ $arch == "skylake" ]
then
    batchsize=(1 8)
    if [ $hyperthreading == 1 ]
    then
        export TJ_NP=32
        export TJ_MPI_MAP_BY="hwthread"
    else
        export TJ_NP=16
        export TJ_MPI_MAP_BY="core"
    fi
else
    batchsize=(1 4)
    if [ $hyperthreading == 1 ]
    then
        export TJ_NP=32
        export TJ_MPI_MAP_BY="hwthread"
    else
        export TJ_NP=4
        export TJ_MPI_MAP_BY="core"
    fi
fi
mesh=('tri')
# mesh=('tet')
form=('helmholtz')
# form=('helmholtz')
vs=('cross-element')  # vectorization strategy
# vs=('omp')
export PYOP2_EXTRA_INFO=1  # switch on timing mode

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
                    export MPICH_CC=$comp
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
