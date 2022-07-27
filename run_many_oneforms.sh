#!/bin/bash
arch='haswell-on-pex'
hyperthreading=1
compiler=('gcc' 'clang')
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
elif [ $arch == "haswell-on-pex" ]
then
    batchsize=(1)  # 1: not vectorize, 4: vectorize by 4
    if [ $hyperthreading == 1 ]
    then
        export TJ_NP=2  # number of processes
        export TJ_MPI_MAP_BY="hwthread"
    else
        export TJ_NP=2
        export TJ_MPI_MAP_BY="core"
    fi
else
    batchsize=(1 8)
    if [ $hyperthreading == 1 ]
    then
        export TJ_NP=32
        export TJ_MPI_MAP_BY="hwthread"
    else
        export TJ_NP=16
        export TJ_MPI_MAP_BY="core"
    fi
fi
mesh=('tri')
form=('inner_schur')
vs=('novect')  # vectorization strategy
opts=("MOP" "FOP")
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
                    for op in ${opts[@]}
                    do
                        export TJ_FORM=$f
                        export TJ_MESH=$m
                        export SV_OPTS=$op
                        export PYOP2_SIMD_WIDTH=$bs
                        if [ $v == 'novect' ]
                        then
                            export PYOP2_VECT_STRATEGY=""
                        else
                            export PYOP2_VECT_STRATEGY=$v
                        fi
                        #export PYOP2_CC=$comp
                        export MPICH_CC=$comp
                        #export OMPI_CC=$comp
                        if [ $comp == "icc" ]
                        then
                            if [ $arch == "haswell" or $arch == "haswell-on-pex" ]
                            then
                                export PYOP2_CFLAGS="-xcore-avx2"
                            else
                                export PYOP2_CFLAGS="-xcore-avx512 -qopt-zmm-usage=high"
                            fi
                        fi
                        firedrake-clean
                        python run_oneforms.py --prefix "$arch"_ --suffix "_$comp"
                        python run_oneforms.py --prefix "$arch"_ --suffix "_$comp"
                    done
                done
            done
        done
    done
done
