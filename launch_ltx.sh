#!/bin/bash
# ============================================================================
# ComfyUI Launch Script — Optimized for RTX 5060 Ti + LTX 2.3 Video
# ============================================================================
#
# GREYOUT PREVENTION PROTOCOL:
#   --bf16-vae        Forces BF16 precision on VAE decode, preventing NaN
#                     overflow that causes "Latent Sludge" on Blackwell GPUs
#
# MEMORY MANAGEMENT:
#   --reserve-vram 6  Reserves 6GB VRAM for display driver + system stability
#                     Prevents UI freezes during large VAE decode passes
#
# WORKFLOW CONSTRAINTS (for LTX 2.3 on 16GB):
#   Resolution:  Width & Height must be divisible by 32
#                (e.g., 768x512, 1024x576, 832x480)
#   Frame Count: Must follow 8n + 1 formula
#                (e.g., 33, 49, 65, 81, 97, 121, 161)
#
# IMPORTANT:
#   - Do NOT install xformers — it will downgrade PyTorch and break cu132
#   - Use fp8 quantized transformer models to fit within 16GB VRAM
#   - Use ONLY the dedicated LTX-Video VAE (not SDXL/SD1.5 VAEs)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Verify CUDA is accessible
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'✓ GPU: {torch.cuda.get_device_name(0)}  |  PyTorch: {torch.__version__}  |  CUDA: {torch.version.cuda}')"
if [ $? -ne 0 ]; then
    echo "ERROR: CUDA/GPU verification failed. Check your NVIDIA drivers."
    exit 1
fi

# Pre-launch memory check
AVAIL_RAM_GB=$(awk '/MemAvailable/ {printf "%.0f", $2/1048576}' /proc/meminfo)
SWAP_FREE_GB=$(awk '/SwapFree/ {printf "%.0f", $2/1048576}' /proc/meminfo)
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   ComfyUI — Blackwell Optimized (RTX 5060 Ti / LTX 2.3)   ║"
echo "║   BF16 VAE: ENABLED | Low VRAM Mode | Reserve: 6GB        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║   RAM Available: ${AVAIL_RAM_GB}GB  |  Swap Free: ${SWAP_FREE_GB}GB                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
if [ "$AVAIL_RAM_GB" -lt 16 ]; then
    echo "⚠ WARNING: Less than 16GB RAM available. Model loading may be slow."
    echo "  Consider closing other applications or increasing swap."
    echo ""
fi

# Launch ComfyUI with Blackwell-safe flags
# --lowvram: Loads model components sequentially to prevent OOM on 32GB systems
# --bf16-vae: Prevents NaN overflow / Latent Sludge on Blackwell GPUs
# --reserve-vram: Keeps 6GB VRAM free for display driver stability
python3 "${SCRIPT_DIR}/main.py" \
    --bf16-vae \
    --lowvram \
    --reserve-vram 3 \
    "$@"
