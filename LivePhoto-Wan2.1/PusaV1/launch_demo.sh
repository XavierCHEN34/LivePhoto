#!/bin/bash

# Pusa-Video V1.0 Demo Launcher Script
# ====================================

echo "🚀 Pusa-Video V1.0 Demo Launcher"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "examples/pusavideo/gradio_demo.py" ]; then
    echo "❌ Error: Please run this script from the PusaV1 directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Warning: No conda environment detected"
    echo "Please activate the pusav1 environment:"
    echo "  conda activate pusav1"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if required models exist
if [ ! -d "model_zoo/PusaV1/Wan2.1-T2V-14B" ]; then
    echo "❌ Base model not found: model_zoo/PusaV1/Wan2.1-T2V-14B"
    echo "Please download the base model first:"
    echo "  huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./model_zoo/PusaV1/Wan2.1-T2V-14B"
    exit 1
fi

if [ ! -f "model_zoo/PusaV1/pusa_v1.pt" ]; then
    echo "❌ Pusa V1.0 model not found: model_zoo/PusaV1/pusa_v1.pt"
    echo "Please download the Pusa V1.0 model first:"
    echo "  huggingface-cli download RaphaelLiu/PusaV1 --local-dir ./model_zoo/PusaV1"
    exit 1
fi

echo "✅ All checks passed!"
echo ""
echo "🎬 Launching Gradio demo..."
echo "The demo will be available at:"
echo "  - Local: http://localhost:7860"
echo "  - Public URL will be shown below"
echo ""
echo "Press Ctrl+C to stop the demo"
echo "---------------------------------"

# Launch the demo
python examples/pusavideo/run_demo.py

echo ""
echo "Demo stopped. Thank you for using Pusa-Video V1.0! 🎬✨" 