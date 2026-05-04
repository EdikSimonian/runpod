#!/usr/bin/env bash
# Run INSIDE the Hot Aisle VM after first SSH.
# Pulls the SGLang ROCm image and sets up host paths.
# Idempotent: safe to re-run.

set -euo pipefail

IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:v0.5.10.post1-rocm700-mi30x}"
HF_CACHE_HOST="${HF_CACHE_HOST:-/var/lib/hf-cache}"

echo "=== VM info ==="
echo "  hostname: $(hostname)"
echo "  ROCm:     $(rocm-smi --showproductname 2>/dev/null | head -3 || echo 'rocm-smi unavailable')"
echo "  Docker:   $(docker --version 2>/dev/null || echo 'NOT INSTALLED')"
echo ""

if ! command -v docker >/dev/null 2>&1; then
    echo "Installing docker..."
    sudo apt-get update -y
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
    sudo usermod -aG docker "$(whoami)"
fi

# Use sudo for docker until the user logs out/in to pick up group membership.
# Without this, the very first vm-prep run on a fresh VM hits "permission denied
# on /var/run/docker.sock" because the current shell predates the usermod.
if docker info >/dev/null 2>&1; then
    DOCKER="docker"
else
    echo "  docker group not yet active in this shell — using sudo for docker commands."
    echo "  Log out + back in (or 'newgrp docker') to drop the sudo wrapper."
    DOCKER="sudo docker"
fi

echo "=== Setting up HF cache ==="
sudo mkdir -p "$HF_CACHE_HOST"
sudo chown "$(id -u):$(id -g)" "$HF_CACHE_HOST"
echo "  $HF_CACHE_HOST owned by $(stat -c '%U:%G' "$HF_CACHE_HOST")"

echo ""
echo "=== Pulling SGLang ROCm image ==="
echo "  Image: $IMAGE  (~19 GB compressed, takes 5-15 min on first pull)"
$DOCKER pull "$IMAGE"

echo ""
echo "=== Pre-flight ==="
$DOCKER run --rm "$IMAGE" python3 - <<'PY'
import sys
ok = True
def show(mod, min_version=None):
    global ok
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "unknown")
        print(f"  {mod}: {v}")
        if min_version and v != "unknown":
            cur = tuple(int(x) for x in v.split(".")[:2] if x.isdigit())
            req = tuple(int(x) for x in min_version.split(".")[:2])
            if cur < req:
                print(f"    FAIL: {mod} must be >= {min_version}"); ok = False
    except ImportError as e:
        print(f"  FAIL {mod}: {e}"); ok = False

show("sglang", "0.5")
show("transformers", "4.55")
show("torch")
sys.exit(0 if ok else 1)
PY

echo ""
echo "=========================================="
echo "VM PREP DONE"
echo "  Next: launch a model"
echo "    bash ~/hotaisle/launch-qwen.sh       # 1× MI300X — fallback path"
echo "    bash ~/hotaisle/launch-mistral.sh    # 2× MI300X — primary path (need 2-GPU VM)"
echo "  Then validate:  bash ~/hotaisle/a0-validate.sh"
echo "=========================================="
