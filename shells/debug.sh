#!/bin/bash

OUT_DIR="/work/NBB/share/datasets/wikitext/magatron-wikitext103"
DATA_PATH="$HOME/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"

PREPROCESS_SCRIPT="/work/NBB/yu_mingzhe/Megatron-LM/tools/preprocess_data.py"

apptainer_image=/work/NBB/yu_mingzhe/pytorch_sandbox_2504

# 预处理 train split
apptainer exec --nv \
  --bind /work/NBB/yu_mingzhe:/work/NBB/yu_mingzhe \
  --bind /work/NBB/share/datasets/wikitext/magatron-wikitext103:/work/NBB/share/datasets/wikitext/magatron-wikitext103 \
  $apptainer_image \
  python "$PREPROCESS_SCRIPT" \
  --input  "$OUT_DIR/wikitext103-validation.jsonl" \
  --output-prefix /work/NBB/yu_mingzhe/Megatron-LM/data/my-gpt2-validation \
  --vocab-file    "$DATA_PATH/vocab.json" \
  --merge-file    "$DATA_PATH/merges.txt" \
  --tokenizer-type GPT2BPETokenizer \
  --append-eod \
  --workers 16 \