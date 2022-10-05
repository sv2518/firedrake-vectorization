#!/bin/bash
# run the script with
# run_many_oneforms.sh --matfree
# or run_many_oneforms.sh --vectorization

# Setup the runs for the achitecture
# if you don't run on pex, add another case here
arch='haswell-on-pex'
hyperthreading=1
if [ $arch == "haswell-on-pex" ]
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
fi

# Vectorisation and slate vectorisation do not access the local matfree infrastruture 
# and can therefore be run with less branches required
if [ $1 == "--vectorization" ]
then
    runtypes=("vectorization" "slatevectorization")
elif [ $1 == "--matfree" ]
then
    runtypes=("matfree" "highordermatfree")
fi

for runtype in ${runtypes[@]}
do

    # decide which forms, meshes, compilers and optimisations to run
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
    elif [[ $runtype == "slatevectorization" || $runtype == "vectorization" ]]
    then
        compiler=('gcc' 'clang')
        mesh=('quad' 'tri' 'hex' 'tet')
        form=('mass' 'helmholtz' 'laplacian' 'elasticity' 'hyperelasticity')
        vs=('cross-element' 'novect')  # vectorization strategy
        opts=("NOP")  # no opts
    fi

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
                            export PYOP2_EXTRA_INFO=1  # switch on timing mode
                            export MPICH_CC=$comp

                            # for the schur complement runs
                            # produce vectorised results only
                            # for highly optimised cased
                            if [ $v == 'novect' ]
                            then
                                batchsize=1
                                export PYOP2_VECT_STRATEGY=""
                            else
                                export PYOP2_VECT_STRATEGY=$v
                                if [[ $runtype != 'vectorization' && $runtype != 'slatevectorization' && $op != 'PFOP' ]]
                                then
                                    break
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
done
