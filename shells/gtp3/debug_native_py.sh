#!/bin/bash

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

WORKSPACE=/work/NBB/yu_mingzhe/Megatron-LM
script_path="$WORKSPACE/pretrain_gpt.py"

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config

CHECKPOINT_PATH="$WORKSPACE/output/checkpoints" #<Specify path>
TENSORBOARD_LOGS_PATH="$WORKSPACE/output/logs" #<Specify path>
VOCAB_FILE="$WORKSPACE/data/vocab.json" #<Specify path to file>/gpt2-vocab.json
MERGE_FILE="$WORKSPACE/data/merges.txt" #<Specify path to file>/gpt2-merges.txt
TRAIN_DATA_PATH="$WORKSPACE/data/wikitext103_train" #<Specify path and file prefix>_text_document
VAL_DATA_PATH="$WORKSPACE/data/wikitext103_validation" #<Specify path and file prefix>_text_document


# GPT_MODEL_ARGS=(
#     --num-layers 96 
#     --hidden-size 12288 
#     --num-attention-heads 96 
#     --seq-length 2048 
#     --max-position-embeddings 2048 
#     --attention-backend auto # Can use (flash/fused/unfused/local)
# )

GPT_MODEL_ARGS=(
       --num-layers 6
       --hidden-size 256
       --num-attention-heads 8 
       --seq-length 1024 
       --max-position-embeddings 1024
       --transformer-impl "local"
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 256
    --train-iters 3
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 50
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --train-data-path $TRAIN_DATA_PATH
    --valid-data-path $VAL_DATA_PATH
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 2
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

# Run the script with MPI
mpirun ${NQSII_MPIOPTS} --mca mpi_abort_print_stack 1 \
  -x PATH -x MASTER_ADDR -x MASTER_PORT -x WORLD_SIZE -x NODE_RANK -x CUDA_DEVICE_MAX_CONNECTIONS \
  -np $WORLD_SIZE --map-by ppr:$NPROC_PER_NODE:node --report-bindings \
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