#!/usr/bin/env python3
"""
Quick test script for Score Vision Miner (skips TensorRT compilation)
Use this for fast development and testing without waiting for engine compilation.
"""

import os
import sys
from pathlib import Path

# Set environment variables for quick testing
os.environ["FORCE_SPEED_MODE"] = "true"  # Force speed mode
os.environ["SKIP_TENSORRT"] = "true"     # Skip TensorRT compilation

# Add miner directory to path
miner_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, miner_dir)

# Import and run the main test
from scripts.test_pipeline import main
import asyncio

if __name__ == "__main__":
    print("🚀 Running quick test (skipping TensorRT compilation)...")
    asyncio.run(main()) 