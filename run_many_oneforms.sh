#!/bin/bash
arch='haswell-on-pex'
hyperthreading=1
if [ $arch == "haswell" ]
then
    batchsize=(4)  # if vectorised else 
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
    # note that haswell on pex is a dual-socket machine
    batchsize=(4)  # if vectorised else 
    if [ $hyperthreading == 1 ]
    then
        export TJ_NP=32  # number of processes
        export TJ_MPI_MAP_BY="hwthread"
    else
        export TJ_NP=16
        export TJ_MPI_MAP_BY="core"
    fi
else
    batchsize=(8)  # if vectorised else 
    if [ $hyperthreading == 1 ]
    then
        export TJ_NP=32
        export TJ_MPI_MAP_BY="hwthread"
    else
        export TJ_NP=16
        export TJ_MPI_MAP_BY="core"
    fi
fi

runtype="highordermatfree"  # or "vectorization" or "matfree"
if [ $runtype == "highordermatfree" ]
then
    compiler=('gcc')
    mesh=('hex')
    form=('inner_schur' 'outer_schur')
    vs=('cross-element' 'novect')  # vectorization strategy
    opts=("PFOP")  # only run for highly optimised case
elif [ $runtype == "matfree" ]
then
    compiler=('gcc')
    mesh=('hex')
    form=('inner_schur' 'outer_schur')
    vs=('cross-element' 'novect')  # vectorization strategy
    opts=("NOP" "MOP" "FOP" "PFOP")  # no opts, resorting, matfree, preconditoned matfree
elif [ $runtype == "slatevectorization" ]
then
    compiler=('gcc' 'clang')
    mesh=('quad' 'tri' 'hex' 'tet')
    form=('mass' 'helmholtz' 'laplacian' 'elasticity' 'hyperelasticity')
    vs=('cross-element' 'novect')  # vectorization strategy
    opts=("NOP")  # no opts
elif [ $runtype == "vectorization" ]
then
    compiler=('gcc' 'clang')
    mesh=('quad' 'tri' 'hex' 'tet')
    form=('mass' 'helmholtz' 'laplacian' 'elasticity' 'hyperelasticity')
    vs=('cross-element' 'novect')  # vectorization strategy
    opts=("NOP")  # no opts
fi
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
			    batchsize=1
                            export PYOP2_VECT_STRATEGY=""
                        else
                            export PYOP2_VECT_STRATEGY=$v
			    # for the schur complement runs
			    # produce vectorised results only
			    # for highly optimised cased
			    if [ $runtype != 'vectorization' and \
				  $runtype != 'slatevectorization' and \
				  $op != 'PFOP' ]
			    then
				break
			    fi
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
                        python run_oneforms.py --prefix "$arch"_ --suffix "_$comp" --runtype "$runtype"
                        python run_oneforms.py --prefix "$arch"_ --suffix "_$comp" --runtype "$runtype"
                    done
                done
            done
        done
    done
done
