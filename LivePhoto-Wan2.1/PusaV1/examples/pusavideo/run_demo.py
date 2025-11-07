#!/usr/bin/env python3
"""
Pusa-Video V1.0 Gradio Demo Launcher
=====================================

This script launches the Gradio demo for Pusa-Video V1.0.
Make sure you have installed all dependencies and downloaded the required models.

Usage:
    python run_demo.py

Requirements:
    - All dependencies installed (see README.md)
    - Models downloaded to ./model_zoo/PusaV1/
    - CUDA-capable GPU (recommended)
"""

import sys
import os
import subprocess
import importlib

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'gradio',
        'torch',
        'diffsynth',
        'PIL',
        'cv2',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'cv2':
                importlib.import_module('cv2')
            else:
                importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nPlease install missing packages:")
        print("pip install gradio torch opencv-python pillow numpy")
        print("pip install -e . # for diffsynth")
        return False
    
    print("✅ All required packages are installed!")
    return True

def check_models():
    """Check if required models are downloaded"""
    base_model_dir = "model_zoo/PusaV1/Wan2.1-T2V-14B"
    lora_path = "model_zoo/PusaV1/pusa_v1.pt"
    
    if not os.path.exists(base_model_dir):
        print(f"❌ Base model directory not found: {base_model_dir}")
        print("\nPlease download the base model:")
        print("huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./model_zoo/PusaV1/Wan2.1-T2V-14B")
        return False
        
    if not os.path.exists(lora_path):
        print(f"❌ LoRA model not found: {lora_path}")
        print("\nPlease download the Pusa V1.0 model:")
        print("huggingface-cli download RaphaelLiu/PusaV1 --local-dir ./model_zoo/PusaV1")
        return False
    
    print("✅ All required models are available!")
    return True

def main():
    try:
        print("🚀 Pusa-Video V1.0 Demo Launcher")
        print("=" * 40)
        
        # Check current directory
        if not os.path.exists("examples/pusavideo/gradio_demo.py"):
            print("❌ Please run this script from the PusaV1 directory")
            print("Current directory:", os.getcwd())
            sys.exit(1)
        
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)
        
        # Check models
        if not check_models():
            sys.exit(1)
        
        # Launch demo
        print("\n[1/4] 🎬 All checks passed. Preparing to launch Gradio demo...")
        
        # Add current directory to path for import
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        print("[2/4] 🐍 Python path updated. Importing demo components...")
        
        from gradio_demo import create_demo
        print("[3/4] ✅ UI components imported successfully. Creating Gradio interface...")
        
        demo = create_demo()
        print("[4/4] ✅ Gradio interface created. Launching server...")
        print(f"   [INFO] Launcher Process ID (PID): {os.getpid()}")
        
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True
        )
        
    except ImportError as e:
        print(f"❌ FATAL: Error importing demo components: {e}")
        print("Please ensure all dependencies are installed correctly in your environment.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"❌ FATAL: An unexpected error occurred during launch: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 