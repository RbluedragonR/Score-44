#!/usr/bin/env python3
"""
Performance Optimizer for Score Vision Miner
Tests different configurations to find optimal speed/quality balance
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import argparse
from loguru import logger

# Add miner directory to path
import sys
miner_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, miner_dir)

from utils.model_manager import ModelManager
from utils.video_processor import VideoProcessor
from sports.annotators.soccer import SoccerAnnotator
from rtx4090_config import (
    get_speed_mode_config, 
    get_quality_mode_config, 
    get_adaptive_config,
    RTX4090PerformanceMonitor
)

class PerformanceOptimizer:
    """Optimize miner performance by testing different configurations."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.results = []
        
    async def test_configuration(self, config_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test a specific configuration and measure performance."""
        logger.info(f"Testing configuration: {config_name}")
        
        # Initialize components with configuration
        model_manager = ModelManager(device="cuda")
        video_processor = VideoProcessor(
            device="cuda",
            batch_size=config.get("batch_size", 16),
            image_size=config.get("image_size", 1280),
            adaptive_mode=False  # Disable adaptive mode for testing
        )
        
        # Load models
        player_model = model_manager.load_model("player")
        ball_model = model_manager.load_model("ball")
        pitch_model = model_manager.load_model("pitch")
        
        # Initialize annotator
        annotator = SoccerAnnotator(
            player_model=player_model,
            ball_model=ball_model,
            pitch_model=pitch_model,
            device="cuda"
        )
        
        # Performance monitoring
        monitor = RTX4090PerformanceMonitor()
        monitor.start_monitoring()
        
        try:
            # Process video
            start_time = time.time()
            frames_data = {}
            frame_count = 0
            
            async for frame_numbers, frames in video_processor.stream_frames_batched(self.video_path):
                # Process batch
                results = await annotator.process_batch(frames, frame_numbers)
                
                # Store results
                for i, frame_num in enumerate(frame_numbers):
                    if i < len(results):
                        frames_data[frame_num] = results[i]
                        frame_count += 1
            
            processing_time = time.time() - start_time
            
            # Get performance metrics
            metrics = monitor.end_monitoring(frame_count)
            
            # Calculate score estimates
            time_margin = 15.0 - processing_time
            speed_score = max(0.0, min(1.0, time_margin / 15.0))  # Simple speed score
            
            # Estimate quality score (higher for better configurations)
            quality_score = 0.8  # Base quality score
            if config.get("image_size", 1280) >= 1920:
                quality_score += 0.1
            if config.get("confidence_threshold", 0.2) >= 0.25:
                quality_score += 0.05
            if config.get("iou_threshold", 0.4) >= 0.45:
                quality_score += 0.05
            
            # Calculate final score (matching validator formula)
            final_score = (quality_score * 0.65) + (speed_score * 0.35)
            
            result = {
                "config_name": config_name,
                "config": config,
                "processing_time": processing_time,
                "time_margin": time_margin,
                "frame_count": frame_count,
                "fps": frame_count / processing_time if processing_time > 0 else 0,
                "quality_score": quality_score,
                "speed_score": speed_score,
                "final_score": final_score,
                "metrics": metrics
            }
            
            logger.info(f"Configuration {config_name} completed:")
            logger.info(f"  Processing time: {processing_time:.2f}s")
            logger.info(f"  Time margin: {time_margin:.2f}s")
            logger.info(f"  FPS: {result['fps']:.2f}")
            logger.info(f"  Final score: {final_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing configuration {config_name}: {e}")
            return {
                "config_name": config_name,
                "error": str(e),
                "processing_time": float('inf'),
                "final_score": 0.0
            }
        finally:
            # Cleanup
            model_manager.clear_cache()
    
    async def run_optimization(self) -> List[Dict[str, Any]]:
        """Run optimization tests with different configurations."""
        logger.info("Starting performance optimization...")
        
        # Define test configurations
        configurations = {
            "Speed Mode": get_speed_mode_config(),
            "Balanced Mode": get_adaptive_config(12.0),
            "Quality Mode": get_quality_mode_config(),
            "Ultra Speed": {
                "batch_size": 24,
                "image_size": 1024,
                "confidence_threshold": 0.15,
                "iou_threshold": 0.3,
                "target_fps": 30,
            },
            "High Quality": {
                "batch_size": 8,
                "image_size": 1920,
                "confidence_threshold": 0.3,
                "iou_threshold": 0.5,
                "target_fps": 12,
            }
        }
        
        # Test each configuration
        for config_name, config in configurations.items():
            result = await self.test_configuration(config_name, config)
            self.results.append(result)
            
            # Small delay between tests
            await asyncio.sleep(2)
        
        return self.results
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze optimization results and provide recommendations."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Filter out failed tests
        valid_results = [r for r in self.results if "error" not in r]
        
        if not valid_results:
            return {"error": "No valid results to analyze"}
        
        # Find best configurations
        best_speed = max(valid_results, key=lambda x: x.get("speed_score", 0))
        best_quality = max(valid_results, key=lambda x: x.get("quality_score", 0))
        best_overall = max(valid_results, key=lambda x: x.get("final_score", 0))
        
        # Calculate statistics
        avg_processing_time = sum(r["processing_time"] for r in valid_results) / len(valid_results)
        avg_fps = sum(r["fps"] for r in valid_results) / len(valid_results)
        
        analysis = {
            "summary": {
                "total_configurations": len(self.results),
                "valid_configurations": len(valid_results),
                "average_processing_time": avg_processing_time,
                "average_fps": avg_fps,
            },
            "best_configurations": {
                "speed": {
                    "name": best_speed["config_name"],
                    "processing_time": best_speed["processing_time"],
                    "time_margin": best_speed["time_margin"],
                    "fps": best_speed["fps"],
                    "speed_score": best_speed["speed_score"],
                },
                "quality": {
                    "name": best_quality["config_name"],
                    "processing_time": best_quality["processing_time"],
                    "quality_score": best_quality["quality_score"],
                    "fps": best_quality["fps"],
                },
                "overall": {
                    "name": best_overall["config_name"],
                    "processing_time": best_overall["processing_time"],
                    "final_score": best_overall["final_score"],
                    "quality_score": best_overall["quality_score"],
                    "speed_score": best_overall["speed_score"],
                    "fps": best_overall["fps"],
                }
            },
            "recommendations": self._generate_recommendations(valid_results),
            "all_results": valid_results
        }
        
        return analysis
    
    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        # Find configurations that meet time requirements
        safe_configs = [r for r in results if r["time_margin"] > 3.0]
        risky_configs = [r for r in results if 0 < r["time_margin"] <= 3.0]
        
        if safe_configs:
            best_safe = max(safe_configs, key=lambda x: x["final_score"])
            recommendations.append(
                f"RECOMMENDED: Use '{best_safe['config_name']}' configuration "
                f"(Final score: {best_safe['final_score']:.3f}, "
                f"Time margin: {best_safe['time_margin']:.2f}s)"
            )
        
        if risky_configs:
            best_risky = max(risky_configs, key=lambda x: x["final_score"])
            recommendations.append(
                f"RISKY: '{best_risky['config_name']}' has high score "
                f"({best_risky['final_score']:.3f}) but low time margin "
                f"({best_risky['time_margin']:.2f}s)"
            )
        
        # Speed vs Quality trade-off analysis
        speed_configs = [r for r in results if r["fps"] > 20]
        quality_configs = [r for r in results if r["quality_score"] > 0.85]
        
        if speed_configs and quality_configs:
            recommendations.append(
                "Consider adaptive configuration based on video length: "
                "Speed mode for long videos, Quality mode for short videos"
            )
        
        return recommendations
    
    def save_results(self, output_path: str):
        """Save optimization results to file."""
        analysis = self.analyze_results()
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")

async def main():
    """Main function for performance optimization."""
    parser = argparse.ArgumentParser(description="Performance Optimizer for Score Vision Miner")
    parser.add_argument("video_path", help="Path to test video file")
    parser.add_argument("--output", "-o", default="optimization_results.json", 
                       help="Output file for results")
    
    args = parser.parse_args()
    
    if not Path(args.video_path).exists():
        logger.error(f"Video file not found: {args.video_path}")
        return
    
    # Run optimization
    optimizer = PerformanceOptimizer(args.video_path)
    results = await optimizer.run_optimization()
    
    # Analyze and save results
    analysis = optimizer.analyze_results()
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE OPTIMIZATION RESULTS")
    print("="*60)
    
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    print(f"Tested {analysis['summary']['total_configurations']} configurations")
    print(f"Average processing time: {analysis['summary']['average_processing_time']:.2f}s")
    print(f"Average FPS: {analysis['summary']['average_fps']:.2f}")
    
    print("\nBEST CONFIGURATIONS:")
    print(f"  Speed: {analysis['best_configurations']['speed']['name']}")
    print(f"    Processing time: {analysis['best_configurations']['speed']['processing_time']:.2f}s")
    print(f"    Time margin: {analysis['best_configurations']['speed']['time_margin']:.2f}s")
    print(f"    FPS: {analysis['best_configurations']['speed']['fps']:.2f}")
    
    print(f"  Quality: {analysis['best_configurations']['quality']['name']}")
    print(f"    Processing time: {analysis['best_configurations']['quality']['processing_time']:.2f}s")
    print(f"    Quality score: {analysis['best_configurations']['quality']['quality_score']:.3f}")
    
    print(f"  Overall: {analysis['best_configurations']['overall']['name']}")
    print(f"    Final score: {analysis['best_configurations']['overall']['final_score']:.3f}")
    print(f"    Processing time: {analysis['best_configurations']['overall']['processing_time']:.2f}s")
    
    print("\nRECOMMENDATIONS:")
    for rec in analysis['recommendations']:
        print(f"  • {rec}")
    
    # Save results
    optimizer.save_results(args.output)
    print(f"\nDetailed results saved to: {args.output}")

if __name__ == "__main__":
    asyncio.run(main()) 