#!/usr/bin/env bash
# Fetch the read-only upstream checkouts into references/.
#
# They are NOT submodules and NOT tracked: each has its own history, we never commit to
# them, and pinning a submodule SHA would imply a version relationship this project does
# not have. They are documentation you can grep.
#
# SSH only — the org's repositories are cloned over SSH, and an HTTPS remote prompts or
# fails depending on the machine.
set -euo pipefail

ORG="git@github.com:ShipControlPrj"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/references"

REPOS=(
  bitbucket-subfaceid                        # the spec: docs/flow.md, docs/new-system-architecture.md
  bitbucket-generic-object-detection-trt     # TensorRT detector + the imgproc CUDA kernels
  bitbucket-generic-feature-extractor-trt    # TensorRT ReID embedder
  gitea-generic-multi-object-tracking-cpp    # DeepSORT: Kalman, Hungarian, lapjv
  bitbucket-motservice                       # single-camera tracking service
  bitbucket-mtmcservice                      # multi-camera tracking service
  bitbucket-countingservice                  # the one Python service — the house style
)

mkdir -p "$DEST"
for repo in "${REPOS[@]}"; do
  target="$DEST/$repo"
  if [ -d "$target/.git" ]; then
    echo "== updating $repo"
    git -C "$target" fetch --depth 1 origin && git -C "$target" reset --hard @{u} || \
      echo "   (could not update $repo; leaving the local checkout alone)"
  else
    echo "== cloning $repo"
    git clone --depth 1 "$ORG/$repo.git" "$target"
  fi
done

echo
echo "references/ is gitignored on purpose — read them, never edit them."
