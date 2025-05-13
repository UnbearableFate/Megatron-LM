#!/bin/bash

#PBS -A NBB
#PBS -q gpu
#PBS -T openmpi
#PBS -b 2
#PBS -l elapstim_req=00:20:00
#PBS -v NQSV_MPI_VER=5.0.7/gcc11.4.0-cuda12.8.1
#PBS -M kanakawapanman@gmail.com

module load openmpi/5.0.7/gcc11.4.0-cuda12.8.1

timestamp=$(date "+%Y%m%d%H%M%S")
apptainer_image=/work/NBB/yu_mingzhe/pytorch_sandbox_2504

mpirun ${NQSII_MPIOPTS} --mca mpi_abort_print_stack 1 \
 -x PATH -np 2 --map-by ppr:1:node --report-bindings \
 apptainer exec --nv \
 --bind /work/NBB/yu_mingzhe:/work/NBB/yu_mingzhe \
 $apptainer_image \
 python /work/NBB/yu_mingzhe/Megatron-LM/run_simple_mcore_train_loop.py \
 --timestamp $timestamp