#!/usr/bin/env bash
set -euo pipefail

cat models/bge-m3/pytorch_model.bin.part-* > models/bge-m3/pytorch_model.bin
cat models/bge-reranker-v2-m3/model.safetensors.part-* > models/bge-reranker-v2-m3/model.safetensors

echo "Reconstructed model files."
