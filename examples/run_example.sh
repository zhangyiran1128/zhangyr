#!/usr/bin/env bash
set -euo pipefail

python scripts/generate_music.py \
  --config configs/train_vhcmusic.yaml \
  --prompt examples/example_prompt.json \
  --checkpoint outputs/checkpoints/diffusion/seed_16/best.pt \
  --bars 32 \
  --output outputs/example \
  --no-render

python scripts/verify_music.py \
  --config configs/train_vhcmusic.yaml \
  --midi outputs/example/generated.mid \
  --plan outputs/example/hierarchical_plan.json \
  --output outputs/example/verify
