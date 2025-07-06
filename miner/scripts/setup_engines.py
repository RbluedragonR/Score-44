#!/usr/bin/env python3
"""
Pre-compile TensorRT engines for Score Vision Miner
This script should be run once to compile all engines, avoiding runtime delays.
"""

import os
import sys
import time
from pathlib import Path
from loguru import logger

# Add miner directory to path
miner_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, miner_dir)

from utils.model_manager import ModelManager

def precompile_engines():
    """Pre-compile all TensorRT engines."""
    logger.info("🚀 Starting TensorRT engine pre-compilation...")
    
    # Check CUDA availability
    import torch
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Cannot compile TensorRT engines.")
        return False
    
    total_start = time.time()
    
    try:
        # Initialize model manager
        model_manager = ModelManager(device="cuda", input_size=(1280, 704))
        
        # Pre-compile player detection engine
        logger.info("Pre-compiling player detection engine...")
        start_time = time.time()
        player_model = model_manager.load_model('player')
        player_time = time.time() - start_time
        logger.info(f"✅ Player detection engine compiled in {player_time:.2f}s")
        
        # Pre-compile pitch detection engine
        logger.info("Pre-compiling pitch detection engine...")
        start_time = time.time()
        pitch_model = model_manager.load_model('pitch')
        pitch_time = time.time() - start_time
        logger.info(f"✅ Pitch detection engine compiled in {pitch_time:.2f}s")
        
        total_time = time.time() - total_start
        logger.info(f"🎉 All engines pre-compiled successfully in {total_time:.2f}s")
        
        # List compiled engines
        data_dir = Path(__file__).parent.parent / "data"
        engine_files = list(data_dir.glob("*.engine"))
        logger.info(f"Compiled engines: {[f.name for f in engine_files]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Engine compilation failed: {e}")
        return False
    finally:
        # Cleanup
        if 'model_manager' in locals():
            model_manager.clear_cache()

def main():
    """Main function."""
    success = precompile_engines()
    
    if success:
        logger.info("\n✅ Engine setup completed successfully!")
        logger.info("You can now run the miner without TensorRT compilation delays.")
    else:
        logger.error("\n❌ Engine setup failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 