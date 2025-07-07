#!/usr/bin/env python3
"""
RTX 4090 Optimization Script for Score Vision Miner
Tests and optimizes configurations for maximum performance and GPU utilization.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from loguru import logger
import torch
import numpy as np

# Add miner directory to path
miner_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, miner_dir)

from utils.model_manager import ModelManager
from utils.video_processor import VideoProcessor
from utils.video_downloader import download_video
from endpoints.soccer import process_soccer_video_rtx4090
from scripts.download_models import download_models

# Test configurations to evaluate
OPTIMIZATION_CONFIGS = {
    "ultra_speed": {
        "batch_size": 24,
        "image_size": 1024,
        "confidence_threshold": 0.1,
        "iou_threshold": 0.3,
        "max_concurrent_batches": 3,
        "description": "Maximum speed mode - lowest latency"
    },
    "high_speed": {
        "batch_size": 20,
        "image_size": 1024,
        "confidence_threshold": 0.15,
        "iou_threshold": 0.35,
        "max_concurrent_batches": 2,
        "description": "High speed mode - balanced speed/accuracy"
    },
    "balanced": {
        "batch_size": 16,
        "image_size": 1280,
        "confidence_threshold": 0.2,
        "iou_threshold": 0.4,
        "max_concurrent_batches": 2,
        "description": "Balanced mode - good speed and accuracy"
    },
    "quality": {
        "batch_size": 12,
        "image_size": 1280,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_concurrent_batches": 1,
        "description": "Quality mode - higher accuracy"
    },
    "max_quality": {
        "batch_size": 8,
        "image_size": 1920,
        "confidence_threshold": 0.3,
        "iou_threshold": 0.5,
        "max_concurrent_batches": 1,
        "description": "Maximum quality mode - highest accuracy"
    }
}

TEST_VIDEO_URL = "https://pub-a55bd0dbae3c4afd86bd066961ab7d1e.r2.dev/test_10secs.mov"

class RTX4090Optimizer:
    """Optimize RTX 4090 settings for maximum performance."""
    
    def __init__(self):
        self.results = []
        self.best_config = None
        
    async def test_configuration(self, config_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test a specific configuration and measure performance."""
        logger.info(f"Testing configuration: {config_name}")
        logger.info(f"Config: {config}")
        
        # Set environment variables for this test
        os.environ["FORCE_SPEED_MODE"] = "true"
        os.environ["BATCH_SIZE"] = str(config["batch_size"])
        os.environ["IMAGE_SIZE"] = str(config["image_size"])
        
        try:
            # Download test video
            video_path = await download_video(TEST_VIDEO_URL)
            
            # Initialize model manager with test configuration
            model_manager = ModelManager(device="cuda", input_size=(config["image_size"], config["image_size"]))
            
            # Load models
            model_manager.load_all_models()
            
            # Measure GPU memory before processing
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            
            # Process video and measure time
            start_time = time.time()
            result = await process_soccer_video_rtx4090(str(video_path), model_manager)
            processing_time = time.time() - start_time
            
            # Measure GPU memory after processing
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            memory_used = peak_memory - initial_memory
            
            # Calculate metrics
            total_frames = len(result["frames"])
            fps = total_frames / processing_time if processing_time > 0 else 0
            time_margin = 15.0 - processing_time
            
            # Estimate validator score
            speed_score = max(0.0, min(1.0, time_margin / 15.0))
            quality_score = self._estimate_quality_score(config)
            final_score = (quality_score * 0.65) + (speed_score * 0.35)
            
            # GPU utilization estimate
            gpu_utilization = min(100.0, (memory_used / 20000) * 100)  # Rough estimate
            
            test_result = {
                "config_name": config_name,
                "config": config,
                "processing_time": processing_time,
                "time_margin": time_margin,
                "total_frames": total_frames,
                "fps": fps,
                "memory_used_mb": memory_used,
                "gpu_utilization": gpu_utilization,
                "speed_score": speed_score,
                "quality_score": quality_score,
                "final_score": final_score,
                "viable": time_margin > 0,
                "competitive": time_margin > 3.0,
                "description": config["description"]
            }
            
            logger.info(f"Results for {config_name}:")
            logger.info(f"  Processing time: {processing_time:.2f}s")
            logger.info(f"  Time margin: {time_margin:.2f}s")
            logger.info(f"  FPS: {fps:.2f}")
            logger.info(f"  Memory used: {memory_used:.1f} MB")
            logger.info(f"  Final score: {final_score:.3f}")
            logger.info(f"  Viable: {test_result['viable']}")
            logger.info(f"  Competitive: {test_result['competitive']}")
            
            # Cleanup
            model_manager.clear_cache()
            torch.cuda.empty_cache()
            
            try:
                os.unlink(video_path)
            except:
                pass
            
            return test_result
            
        except Exception as e:
            logger.error(f"Error testing configuration {config_name}: {e}")
            return {
                "config_name": config_name,
                "error": str(e),
                "processing_time": float('inf'),
                "final_score": 0.0,
                "viable": False,
                "competitive": False
            }
    
    def _estimate_quality_score(self, config: Dict[str, Any]) -> float:
        """Estimate quality score based on configuration parameters."""
        base_score = 0.7
        
        # Image size bonus
        if config["image_size"] >= 1920:
            base_score += 0.15
        elif config["image_size"] >= 1280:
            base_score += 0.1
        elif config["image_size"] >= 1024:
            base_score += 0.05
        
        # Confidence threshold bonus
        if config["confidence_threshold"] >= 0.25:
            base_score += 0.1
        elif config["confidence_threshold"] >= 0.2:
            base_score += 0.05
        
        # IOU threshold bonus
        if config["iou_threshold"] >= 0.45:
            base_score += 0.05
        
        return min(1.0, base_score)
    
    async def run_optimization(self) -> List[Dict[str, Any]]:
        """Run optimization tests with all configurations."""
        logger.info("🚀 Starting RTX 4090 optimization tests...")
        
        # Ensure models are downloaded
        download_models()
        
        # Test each configuration
        for config_name, config in OPTIMIZATION_CONFIGS.items():
            logger.info(f"\n--- Testing {config_name} ---")
            result = await self.test_configuration(config_name, config)
            self.results.append(result)
            
            # Small delay between tests
            await asyncio.sleep(2)
        
        return self.results
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze optimization results and provide recommendations."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Filter valid results
        valid_results = [r for r in self.results if "error" not in r]
        
        if not valid_results:
            return {"error": "No valid results to analyze"}
        
        # Find best configurations
        viable_results = [r for r in valid_results if r["viable"]]
        competitive_results = [r for r in valid_results if r["competitive"]]
        
        best_speed = max(valid_results, key=lambda x: x.get("fps", 0))
        best_score = max(valid_results, key=lambda x: x.get("final_score", 0))
        fastest_viable = min(viable_results, key=lambda x: x["processing_time"]) if viable_results else None
        
        analysis = {
            "summary": {
                "total_configs": len(self.results),
                "valid_configs": len(valid_results),
                "viable_configs": len(viable_results),
                "competitive_configs": len(competitive_results),
            },
            "best_configurations": {
                "fastest": {
                    "name": best_speed["config_name"],
                    "fps": best_speed["fps"],
                    "processing_time": best_speed["processing_time"],
                    "viable": best_speed["viable"]
                },
                "highest_score": {
                    "name": best_score["config_name"],
                    "final_score": best_score["final_score"],
                    "processing_time": best_score["processing_time"],
                    "viable": best_score["viable"]
                },
                "fastest_viable": fastest_viable if fastest_viable else None
            },
            "recommendations": self._generate_recommendations(valid_results, viable_results, competitive_results),
            "all_results": valid_results
        }
        
        return analysis
    
    def _generate_recommendations(self, valid_results: List[Dict], viable_results: List[Dict], competitive_results: List[Dict]) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        if not viable_results:
            recommendations.append("❌ CRITICAL: No configurations complete within 15s limit!")
            recommendations.append("🔧 Try reducing batch size, image size, or enable more aggressive optimizations")
            return recommendations
        
        if competitive_results:
            best_competitive = max(competitive_results, key=lambda x: x["final_score"])
            recommendations.append(
                f"🏆 RECOMMENDED: Use '{best_competitive['config_name']}' configuration"
            )
            recommendations.append(
                f"   Final score: {best_competitive['final_score']:.3f}, "
                f"Time margin: {best_competitive['time_margin']:.2f}s"
            )
        
        fastest_viable = min(viable_results, key=lambda x: x["processing_time"])
        if fastest_viable["time_margin"] < 3.0:
            recommendations.append(
                f"⚠️ RISKY: Fastest viable config '{fastest_viable['config_name']}' "
                f"has low time margin ({fastest_viable['time_margin']:.2f}s)"
            )
        
        # GPU utilization recommendations
        high_gpu_configs = [r for r in valid_results if r.get("gpu_utilization", 0) > 80]
        if high_gpu_configs:
            recommendations.append(
                f"🚀 GPU UTILIZATION: {len(high_gpu_configs)} configs use >80% GPU - good utilization"
            )
        else:
            recommendations.append(
                "⚠️ GPU UTILIZATION: Consider increasing batch size for better GPU utilization"
            )
        
        return recommendations
    
    def save_results(self, output_path: str):
        """Save optimization results to file."""
        analysis = self.analyze_results()
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
    
    def print_summary(self):
        """Print optimization summary."""
        analysis = self.analyze_results()
        
        print("\n" + "="*70)
        print("RTX 4090 OPTIMIZATION RESULTS")
        print("="*70)
        
        if "error" in analysis:
            print(f"Error: {analysis['error']}")
            return
        
        summary = analysis["summary"]
        print(f"Tested {summary['total_configs']} configurations")
        print(f"Valid: {summary['valid_configs']}, Viable: {summary['viable_configs']}, Competitive: {summary['competitive_configs']}")
        
        print("\n🏆 BEST CONFIGURATIONS:")
        
        best = analysis["best_configurations"]
        print(f"  Fastest: {best['fastest']['name']}")
        print(f"    FPS: {best['fastest']['fps']:.2f}")
        print(f"    Time: {best['fastest']['processing_time']:.2f}s")
        print(f"    Viable: {'✅' if best['fastest']['viable'] else '❌'}")
        
        print(f"  Highest Score: {best['highest_score']['name']}")
        print(f"    Score: {best['highest_score']['final_score']:.3f}")
        print(f"    Time: {best['highest_score']['processing_time']:.2f}s")
        print(f"    Viable: {'✅' if best['highest_score']['viable'] else '❌'}")
        
        if best['fastest_viable']:
            print(f"  Fastest Viable: {best['fastest_viable']['config_name']}")
            print(f"    Time: {best['fastest_viable']['processing_time']:.2f}s")
            print(f"    Margin: {best['fastest_viable']['time_margin']:.2f}s")
        
        print("\n📋 RECOMMENDATIONS:")
        for rec in analysis['recommendations']:
            print(f"  {rec}")
        
        print("\n📊 DETAILED RESULTS:")
        for result in analysis['all_results']:
            status = "✅" if result['viable'] else "❌"
            competitive = "🏆" if result['competitive'] else "⚠️"
            print(f"  {status} {competitive} {result['config_name']}: {result['processing_time']:.2f}s "
                  f"({result['fps']:.1f} FPS, Score: {result['final_score']:.3f})")

async def main():
    """Main optimization function."""
    logger.info("RTX 4090 Optimization Script")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Cannot optimize RTX 4090.")
        return
    
    # Check if RTX 4090 is available
    device_name = torch.cuda.get_device_name(0)
    if "4090" not in device_name:
        logger.warning(f"GPU detected: {device_name} (not RTX 4090)")
        logger.warning("Optimization may not be optimal for your GPU")
    else:
        logger.info(f"RTX 4090 detected: {device_name}")
    
    # Run optimization
    optimizer = RTX4090Optimizer()
    await optimizer.run_optimization()
    
    # Print and save results
    optimizer.print_summary()
    
    output_file = Path(__file__).parent.parent / "rtx4090_optimization_results.json"
    optimizer.save_results(str(output_file))
    
    print(f"\n📁 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main()) 