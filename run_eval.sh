#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

cd /data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper
source /data/vepfs/users/intern/lingyue.yang/openpi/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

uv run --active python scripts/open_loop_eval.py \
  --checkpoint-dir third_party/openpi/checkpoints/pi05_aloha_wbcd_lora/evorl_pi05_lora_fullA_normfix_260319 \
  --repo-id pipeline_ab/A \
  --episodes all \
  --output-dir tmp/open_loop_eval_260319 \
  > open_loop_eval_260319.log 2>&1
