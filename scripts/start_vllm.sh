#!/bin/bash
# Start vLLM server for a specific model from configs/models.yaml
#
# Usage:
#   ./scripts/start_vllm.sh <model_key>
#   ./scripts/start_vllm.sh llama3.1-8b
#   ./scripts/start_vllm.sh mistral-7b
#
# Available models (all 8-bit quantized):
#   glm4-9b      - THUDM GLM-4-9B
#   llama3.1-8b  - Meta Llama 3.1 8B Instruct
#   qwen2.5-vl-7b - Qwen2.5-VL-7B-Instruct
#   mistral-7b   - Mistral 7B Instruct v0.3
#   gemma-2b     - Google Gemma 2B
#   gemma-7b     - Google Gemma 7B
#   tinyllama    - TinyLlama 1.1B
#   phi3-mini    - Microsoft Phi-3 Mini

set -e

MODEL_KEY="${1:-mistral-7b}"

# Model configurations (8-bit quantized)
case "$MODEL_KEY" in
    glm4-9b)
        MODEL_ID="THUDM/glm-4-9b-chat"
        MAX_LEN=8192
        GPU_UTIL=0.85
        ;;
    llama3.1-8b)
        MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
        MAX_LEN=8192
        GPU_UTIL=0.85
        ;;
    qwen2.5-vl-7b)
        MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
        MAX_LEN=8192
        GPU_UTIL=0.85
        ;;
    mistral-7b)
        MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
        MAX_LEN=8192
        GPU_UTIL=0.85
        ;;
    gemma-2b)
        MODEL_ID="google/gemma-2b-it"
        MAX_LEN=8192
        GPU_UTIL=0.70
        ;;
    gemma-7b)
        MODEL_ID="google/gemma-7b-it"
        MAX_LEN=8192
        GPU_UTIL=0.85
        ;;
    tinyllama)
        MODEL_ID="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        MAX_LEN=4096
        GPU_UTIL=0.50
        ;;
    phi3-mini)
        MODEL_ID="microsoft/Phi-3-mini-4k-instruct"
        MAX_LEN=4096
        GPU_UTIL=0.60
        EXTRA_ARGS="--trust-remote-code"
        ;;
    *)
        echo "Unknown model: $MODEL_KEY"
        echo "Available models: glm4-9b, llama3.1-8b, qwen2.5-vl-7b, mistral-7b, gemma-2b, gemma-7b, tinyllama, phi3-mini"
        exit 1
        ;;
esac

echo "Starting vLLM server for: $MODEL_KEY"
echo "  Model ID: $MODEL_ID"
echo "  Quantization: 8-bit (bitsandbytes)"
echo "  Max context: $MAX_LEN"
echo "  GPU utilization: $GPU_UTIL"

vllm serve "$MODEL_ID" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    ${EXTRA_ARGS:-}
