#!/bin/bash
#PBS -A NBB
#PBS -q gpu
#PBS -T openmpi
#PBS -b 8
#PBS -l elapstim_req=00:40:00
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

WORKSPACE=/work/NBB/yu_mingzhe/Megatron-LM
script_path="$WORKSPACE/pretrain_gpt.py"

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config

CHECKPOINT_PATH="$WORKSPACE/output/checkpoints/$timestamp" #<Specify path>
TENSORBOARD_LOGS_PATH="$WORKSPACE/output/logs/$timestamp" #<Specify path>
VOCAB_FILE="$WORKSPACE/data/vocab.json" #<Specify path to file>/gpt2-vocab.json
MERGE_FILE="$WORKSPACE/data/merges.txt" #<Specify path to file>/gpt2-merges.txt
TRAIN_DATA_PATH="$WORKSPACE/data/wikitext103_train" #<Specify path and file prefix>_text_document
VAL_DATA_PATH="$WORKSPACE/data/wikitext103_validation" #<Specify path and file prefix>_text_document

GPT_MODEL_ARGS=(
       --num-layers 12
       --hidden-size 512
       --num-attention-heads 8
       --seq-length 1024
       --max-position-embeddings 1024
       --transformer-impl "local"
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 512
    --train-iters 500
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 8
    --pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --train-data-path $TRAIN_DATA_PATH
    --valid-data-path $VAL_DATA_PATH
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 2
    --save-interval 100
    --eval-interval 10
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 3
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

# Run the script with MPI
mpirun ${NQSII_MPIOPTS} --mca mpi_abort_print_stack 1 \
  -x PATH -x MASTER_ADDR -x MASTER_PORT -x WORLD_SIZE -x NODE_RANK -x CUDA_DEVICE_MAX_CONNECTIONS \
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
      ${GPT_MODEL_ARGS[@]} \
      ${TRAINING_ARGS[@]} \
      ${MODEL_PARALLEL_ARGS[@]} \
      ${DATA_ARGS[@]} \
      ${EVAL_AND_LOGGING_ARGS[@]}