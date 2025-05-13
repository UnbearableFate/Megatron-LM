#!/bin/bash
#PBS -A NBB
#PBS -q gpu
#PBS -T openmpi
#PBS -b 1
#PBS -l elapstim_req=01:00:00
#PBS -v NQSV_MPI_VER=5.0.7/gcc11.4.0-cuda12.8.1
#PBS -M kanakawapanman@gmail.com

module load openmpi/5.0.7/gcc11.4.0-cuda12.8.1

OUT_DIR="/work/NBB/share/datasets/wikitext/magatron-wikitext103"
DATA_PATH="$HOME/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"

PYTHON="/work/NBB/yu_mingzhe/miniconda3/envs/py313/bin/python"
PREPROCESS_SCRIPT="/work/NBB/yu_mingzhe/Megatron-LM/tools/preprocess_data.py"

# 预处理 train split
"$PYTHON" "$PREPROCESS_SCRIPT" \
  --input  "$OUT_DIR/wikitext103-train.jsonl" \
  --output-prefix my-gpt2 \
  --vocab-file    "$DATA_PATH/vocab.json" \
  --merge-file    "$DATA_PATH/merges.txt" \
  --tokenizer-type GPT2BPETokenizer \
  --append-eod \
  --workers 16 \

# 预处理 validation split
"$PYTHON" "$PREPROCESS_SCRIPT" \
  --input  "$OUT_DIR/wikitext103-validation.jsonl" \
  --output-prefix my-gpt2 \
  --vocab-file    "$DATA_PATH/vocab.json" \
  --merge-file    "$DATA_PATH/merges.txt" \
  --tokenizer-type GPT2BPETokenizer \
  --append-eod