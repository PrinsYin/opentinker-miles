#!/bin/bash
# OpenTinker-Miles All-in-One Entrypoint
#
# This script:
# 1. Creates required data directories
# 2. Starts Ray head node
# 3. Starts the OpenTinker training API
#
# Environment variables:
#   NUM_GPUS: Number of GPUs for Ray (default: auto-detect)
#   TRAINING_PORT: Port for training API (default: 8000)
#   RAY_DASHBOARD_PORT: Port for Ray dashboard (default: 8265)
#   RAY_CLIENT_PORT: Port for Ray client (default: 10001)
#   SKIP_RAY: Set to "1" to skip Ray startup (for connecting to external Ray)

set -e

echo "========================================"
echo "OpenTinker-Miles Starting..."
echo "========================================"

# Create data directories (mounted from host or emptyDir)
echo "Creating data directories..."
mkdir -p /data/models /data/checkpoints /data/datasets /data/trajectories /data/metadata
chmod -R 777 /data 2>/dev/null || true

# Ensure gsm8k_rl.jsonl exists (required by RolloutManager)
# This file is needed even if /data is mounted from host
if [ ! -f /data/datasets/gsm8k_rl.jsonl ]; then
    echo "Creating gsm8k_rl.jsonl dataset file..."
    # Try to copy from image location first (if it exists)
    if [ -f /app/training/test_data/gsm8k_rl.jsonl ]; then
        cp /app/training/test_data/gsm8k_rl.jsonl /data/datasets/gsm8k_rl.jsonl
        echo "✓ Copied gsm8k_rl.jsonl from image"
    else
        # Create a minimal valid JSONL file as fallback
        cat > /data/datasets/gsm8k_rl.jsonl << 'EOF'
{"prompt": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", "response": "16 - 3 - 4 = <<16-3-4=9>>9\nShe makes 9 * 2 = $<<9*2=18>>18 every day.\n#### 18"}
{"prompt": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "response": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3"}
{"prompt": "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?", "response": "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000"}
{"prompt": "James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?", "response": "He sprints 3*3=<<3*3=9>>9 times\nSo he runs 9*60=<<9*60=540>>540 meters\n#### 540"}
{"prompt": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. How many cups of feed does she need in the morning?", "response": "If each chicken eats 3 cups of feed per day, and there are 3 meals, then each chicken gets 3/3=<<3/3=1>>1 cup of feed per meal.\nSince this is asked about the morning meal, the answer is 1 cup.\n#### 1"}
EOF
        echo "✓ Created minimal gsm8k_rl.jsonl file"
    fi
fi

# Prepare data: download models and datasets if not already present
if [ -f /prepare_data.sh ]; then
    echo "Preparing data..."
    /prepare_data.sh || {
        echo "WARNING: Data preparation failed."
    }
fi

# Convert model to Megatron format if needed (requires GPU)
if [ -f /convert_model.sh ]; then
    echo "Checking if model conversion is needed..."
    /convert_model.sh || {
        echo "WARNING: Model conversion failed. This might be expected if no GPU is available."
        echo "The conversion will need to be done manually when GPU is available."
    }
fi

# Verify Miles is available
echo "Checking Miles installation..."
python -c "import miles; print(f'Miles version: {getattr(miles, \"__version__\", \"installed\")}')" || {
    echo "ERROR: Miles not found in PYTHONPATH"
    exit 1
}

# Start Ray head node (unless SKIP_RAY is set)
if [ "${SKIP_RAY}" != "1" ]; then
    # Auto-detect GPUs if not specified
    if [ -z "${NUM_GPUS}" ]; then
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
        echo "Auto-detected ${NUM_GPUS} GPUs"
    fi

    # Get node IP
    NODE_IP=${MASTER_ADDR:-$(hostname -i)}

    echo "Starting Ray head node..."
    echo "  Node IP: ${NODE_IP}"
    echo "  GPUs: ${NUM_GPUS}"
    echo "  Dashboard: ${RAY_DASHBOARD_PORT:-8265}"
    echo "  Client: ${RAY_CLIENT_PORT:-10001}"

    ray start --head \
        --node-ip-address "${NODE_IP}" \
        --num-gpus "${NUM_GPUS}" \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${RAY_DASHBOARD_PORT:-8265}" \
        --ray-client-server-port="${RAY_CLIENT_PORT:-10001}"

    # Wait for Ray to be ready
    sleep 2
    ray status

    # Set RAY_ADDRESS for the training API
    export RAY_ADDRESS="ray://localhost:${RAY_CLIENT_PORT:-10001}"
else
    echo "SKIP_RAY=1, not starting Ray head node"
    echo "Expecting RAY_ADDRESS to be set externally: ${RAY_ADDRESS}"
fi

echo ""
echo "========================================"
echo "Starting OpenTinker Training API..."
echo "  Host: ${TRAINING_HOST:-0.0.0.0}"
echo "  Port: ${TRAINING_PORT:-8000}"
echo "  Ray: ${RAY_ADDRESS}"
echo "========================================"
echo ""

# Start the training API (foreground)
exec python3 -m training
