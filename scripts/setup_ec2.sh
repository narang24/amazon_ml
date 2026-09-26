#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_ec2.sh — One-time EC2 setup script
# Run this ONCE after launching a fresh Ubuntu EC2 instance.
# Usage: bash setup_ec2.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "=== [1/6] Updating system packages ==="
sudo apt-get update -y
sudo apt-get upgrade -y

echo "=== [2/6] Installing Python 3.11 and system tools ==="
sudo apt-get install -y python3.11 python3.11-venv python3-pip \
    git rsync awscli htop tmux unzip curl build-essential

echo "=== [3/6] Creating project directory ==="
mkdir -p ~/amazon_ml/{data/train,data/test,processed,output,config,mlruns}

echo "=== [4/6] Installing Python dependencies ==="
cd ~/amazon_ml
if [ -f requirements.txt ]; then
    python3.11 -m pip install --upgrade pip
    python3.11 -m pip install -r requirements.txt
else
    echo "[warn] requirements.txt not found — dependencies will be installed on first deploy"
fi

echo "=== [5/6] Configuring AWS CLI ==="
# The EC2 instance should use an IAM role with S3 access.
# Verify:
aws sts get-caller-identity || echo "[warn] AWS credentials not configured. Attach IAM role to EC2."

echo "=== [6/6] Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Upload dataset to S3:  aws s3 cp ./data/ s3://YOUR_BUCKET/entity-resolution/data/ --recursive"
echo "  2. Set GitHub secrets (see README.md)"
echo "  3. Push to main branch to trigger the pipeline"
