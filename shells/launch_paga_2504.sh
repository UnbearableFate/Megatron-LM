#!/bin/bash
#PBS -A NBB
#PBS -q gpu
#PBS -T openmpi
#PBS -b 2
#PBS -l elapstim_req=00:05:00
#PBS -v NQSV_MPI_VER=5.0.7/gcc11.4.0-cuda12.8.1
#PBS -M kanakawapanman@gmail.com

module load openmpi/5.0.7/gcc11.4.0-cuda12.8.1

# compute world size
NNODES=$(sort -u "$PBS_NODEFILE" | wc -l)
NPROC_PER_NODE=1
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# master address and port
MASTER_ADDR=$(head -n 1 "$PBS_NODEFILE")
MASTER_PORT=6003

# compute local node-rank
hosts=( $(sort -u "$PBS_NODEFILE") )
NODE_RANK=0
for idx in "${!hosts[@]}"; do
  if [[ "${hosts[idx]}" == "$(hostname)" ]]; then
    NODE_RANK=$idx
    break
  fi
done

echo "NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE  WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  NODE_RANK=$NODE_RANK"

timestamp=$(date "+%Y%m%d%H%M%S")
apptainer_image=/work/NBB/yu_mingzhe/pytorch_sandbox_2504
script_path=/work/NBB/yu_mingzhe/Megatron-LM/test.py

# Run the script with MPI
mpirun ${NQSII_MPIOPTS} --mca mpi_abort_print_stack 1 \
  -x PATH -x MASTER_ADDR -x MASTER_PORT -x WORLD_SIZE -x NODE_RANK \
  -np $WORLD_SIZE --map-by ppr:$NPROC_PER_NODE:node --report-bindings \
  apptainer exec --nv \
    --bind /work/NBB/yu_mingzhe:/work/NBB/yu_mingzhe \
    $apptainer_image \
    torchrun \
      --rdzv-backend=c10d \
      --rdzv-endpoint="$MASTER_ADDR":"$MASTER_PORT" \
      --nnodes=$NNODES \
      --nproc-per-node=$NPROC_PER_NODE \
      --node-rank=$NODE_RANK \
      $script_path \
      --timestamp $timestamp