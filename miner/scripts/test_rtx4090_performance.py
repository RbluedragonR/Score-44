#!/usr/bin/env python3
"""
RTX 4090 Performance Test Script for Score Vision Mining
Tests and benchmarks the RTX 4090 optimizations
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from loguru import logger
from typing import Dict, Any

# Add miner directory to path
miner_dir = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, miner_dir)

from utils.model_manager import ModelManager
from utils.video_downloader import download_video
from utils.device import optimize_cuda_for_rtx4090, get_rtx4090_optimal_batch_size, get_rtx4090_optimal_image_size
from rtx4090_config import RTX4090_CONFIG, RTX4090PerformanceMonitor, optimize_cuda_settings
from scripts.download_models import download_models

# Test video URL
TEST_VIDEO_URL = "https://pub-a55bd0dbae3c4afd86bd066961ab7d1e.r2.dev/test_10secs.mov"

async def test_rtx4090_optimizations():
    """Test RTX 4090 specific optimizations."""
    logger.info("=== RTX 4090 Optimization Test ===")
    
    # Test CUDA availability and optimization
    import torch
    if torch.cuda.is_available():
        logger.info("✅ CUDA is available")
        
        # Apply RTX 4090 optimizations
        optimize_cuda_settings()
        
        # Check GPU properties
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        logger.info(f"GPU: {props.name}")
        logger.info(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
        logger.info(f"CUDA Capability: {props.major}.{props.minor}")
        
        # Test optimal parameters
        batch_size = get_rtx4090_optimal_batch_size()
        image_size = get_rtx4090_optimal_image_size()
        
        logger.info(f"Optimal batch size: {batch_size}")
        logger.info(f"Optimal image size: {image_size}")
        
        return True
    else:
        logger.error("❌ CUDA is not available")
        return False

async def test_model_loading():
    """Test model loading with RTX 4090 optimizations."""
    logger.info("=== Model Loading Test ===")
    
    try:
        # Initialize model manager
        model_manager = ModelManager(device="cuda")
        
        # Test loading each model
        models_to_test = ["player", "pitch", "ball"]
        
        for model_name in models_to_test:
            start_time = time.time()
            model = model_manager.get_model(model_name)
            load_time = time.time() - start_time
            
            logger.info(f"✅ {model_name} model loaded in {load_time:.2f}s")
        
        return True
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

async def test_batch_processing():
    """Test batch processing performance."""
    logger.info("=== Batch Processing Test ===")
    
    try:
        # Download test video
        logger.info(f"Downloading test video from {TEST_VIDEO_URL}")
        video_path = await download_video(TEST_VIDEO_URL)
        
        # Initialize components
        model_manager = ModelManager(device="cuda")
        player_model = model_manager.get_model("player")
        pitch_model = model_manager.get_model("pitch")
        
        # Test batch processing
        batch_size = RTX4090_CONFIG["batch_size"]
        image_size = RTX4090_CONFIG["image_size"]
        
        logger.info(f"Testing batch processing with batch_size={batch_size}, image_size={image_size}")
        
        # Create dummy batch
        import numpy as np
        dummy_batch = [np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(batch_size)]
        
        # Test player detection
        start_time = time.time()
        player_results = player_model(dummy_batch, imgsz=image_size, verbose=False)
        player_time = time.time() - start_time
        
        # Test pitch detection
        start_time = time.time()
        pitch_results = pitch_model(dummy_batch, verbose=False)
        pitch_time = time.time() - start_time
        
        logger.info(f"✅ Player detection batch: {player_time:.2f}s ({batch_size/player_time:.1f} fps)")
        logger.info(f"✅ Pitch detection batch: {pitch_time:.2f}s ({batch_size/pitch_time:.1f} fps)")
        
        # Cleanup
        try:
            os.unlink(video_path)
        except:
            pass
        
        return True
    except Exception as e:
        logger.error(f"❌ Batch processing test failed: {e}")
        return False

async def test_memory_management():
    """Test GPU memory management."""
    logger.info("=== Memory Management Test ===")
    
    try:
        import torch
        import GPUtil
        
        # Get initial GPU memory
        initial_memory = GPUtil.getGPUs()[0].memoryUsed
        logger.info(f"Initial GPU memory: {initial_memory} MB")
        
        # Allocate some tensors
        tensors = []
        for i in range(5):
            tensor = torch.zeros(1, 3, 1920, 1920, device="cuda")
            tensors.append(tensor)
        
        # Check memory usage
        current_memory = GPUtil.getGPUs()[0].memoryUsed
        logger.info(f"After allocation: {current_memory} MB")
        logger.info(f"Memory used: {current_memory - initial_memory} MB")
        
        # Clear tensors and cache
        del tensors
        torch.cuda.empty_cache()
        
        # Check final memory
        final_memory = GPUtil.getGPUs()[0].memoryUsed
        logger.info(f"After cleanup: {final_memory} MB")
        
        if final_memory <= initial_memory + 100:  # Allow some overhead
            logger.info("✅ Memory management working correctly")
            return True
        else:
            logger.warning("⚠️ Memory cleanup may not be optimal")
            return False
            
    except Exception as e:
        logger.error(f"❌ Memory management test failed: {e}")
        return False

async def run_full_performance_test():
    """Run complete RTX 4090 performance test."""
    logger.info("🚀 Starting RTX 4090 Performance Test Suite")
    
    # Initialize performance monitor
    monitor = RTX4090PerformanceMonitor()
    
    # Run tests
    tests = [
        ("CUDA Optimization", test_rtx4090_optimizations),
        ("Model Loading", test_model_loading),
        ("Batch Processing", test_batch_processing),
        ("Memory Management", test_memory_management),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            start_time = time.time()
            success = await test_func()
            test_time = time.time() - start_time
            
            results[test_name] = {
                "success": success,
                "time": test_time
            }
            
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"{status} {test_name} ({test_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = {
                "success": False,
                "time": 0,
                "error": str(e)
            }
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("RTX 4090 PERFORMANCE TEST SUMMARY")
    logger.info("="*50)
    
    passed = sum(1 for r in results.values() if r["success"])
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        logger.info(f"{status} {test_name}: {result['time']:.2f}s")
        if "error" in result:
            logger.error(f"  Error: {result['error']}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! RTX 4090 is optimized for Score Vision mining.")
    else:
        logger.warning("⚠️ Some tests failed. Check the logs for details.")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_full_performance_test()) 