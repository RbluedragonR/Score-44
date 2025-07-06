import asyncio
import time
from typing import AsyncGenerator, Optional, Tuple, List, Dict, Any
import cv2
import numpy as np
import supervision as sv
from loguru import logger
import torch
from pathlib import Path
import os

# Import RTX 4090 configuration
try:
    from rtx4090_config import get_adaptive_config, get_speed_mode_config, get_quality_mode_config
except ImportError:
    # Fallback configuration if rtx4090_config is not available
    def get_adaptive_config(target_time: float = 12.0) -> Dict[str, Any]:
        return {
            "batch_size": 16,
            "image_size": 1280,
            "confidence_threshold": 0.2,
            "iou_threshold": 0.4,
            "target_fps": 20,
        }

class VideoProcessor:
    """Handles video processing with frame streaming and timeout management optimized for RTX 4090."""
    
    def __init__(
        self,
        device: str = "cpu",
        cuda_timeout: float = 900.0,  # 15 minutes for CUDA
        mps_timeout: float = 1800.0,  # 30 minutes for MPS
        cpu_timeout: float = 10800.0,  # 3 hours for CPU
        batch_size: int = 10,  # RTX 4090 can handle larger batches
        image_size: int = 1920,  # Larger image size for RTX 4090
        adaptive_mode: bool = True,  # Enable adaptive configuration
    ):
        self.device = device
        self.batch_size = batch_size
        self.image_size = image_size
        self.adaptive_mode = adaptive_mode
        
        # Check for force speed mode environment variable
        self.force_speed_mode = os.getenv("FORCE_SPEED_MODE", "false").lower() == "true"
        if self.force_speed_mode:
            logger.info("FORCE_SPEED_MODE enabled - will use speed configuration for all videos")
        
        # Set timeout based on device
        if device == "cuda":
            self.processing_timeout = cuda_timeout
        elif device == "mps":
            self.processing_timeout = mps_timeout
        else:  # cpu or any other device
            self.processing_timeout = cpu_timeout
            
        logger.info(f"Video processor initialized with {device} device, timeout: {self.processing_timeout:.1f}s")
        logger.info(f"RTX 4090 optimizations: batch_size={batch_size}, image_size={image_size}")
        logger.info(f"Adaptive mode: {adaptive_mode}")
        logger.info(f"Force speed mode: {self.force_speed_mode}")
    
    def _get_adaptive_config(self, video_path: str) -> Dict[str, Any]:
        """
        Get adaptive configuration based on video characteristics.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Optimized configuration
        """
        # Force speed mode takes priority
        if self.force_speed_mode:
            logger.info("Using FORCE_SPEED_MODE configuration")
            config = get_speed_mode_config()
            self.batch_size = config.get("batch_size", self.batch_size)
            self.image_size = config.get("image_size", self.image_size)
            return config
        
        if not self.adaptive_mode:
            return {
                "batch_size": self.batch_size,
                "image_size": self.image_size,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
            }
        
        try:
            # Get video info
            video_info = self.get_video_info(video_path)
            total_frames = video_info.total_frames
            fps = video_info.fps
            duration = total_frames / fps if fps > 0 else 0
            
            # Estimate processing time based on video length
            estimated_time = duration * 0.1  # Rough estimate: 10% of video duration
            
            logger.info(f"Video analysis: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
            logger.info(f"Estimated processing time: {estimated_time:.1f}s")
            
            # PRIORITY: Use speed mode for most videos to stay under 15s limit
            # Only use quality mode for very short videos where we have time margin
            if estimated_time <= 5.0 and duration <= 30.0:
                # Very short video - use quality mode
                config = get_quality_mode_config()
                logger.info("Using QUALITY MODE configuration for very short video")
            elif estimated_time <= 8.0:
                # Short video - use balanced mode
                config = get_adaptive_config(12.0)
                logger.info("Using BALANCED MODE configuration for short video")
            else:
                # Medium/long video - use speed mode (PRIORITY)
                config = get_speed_mode_config()
                logger.info("Using SPEED MODE configuration for medium/long video")
            
            # Update instance variables
            self.batch_size = config.get("batch_size", self.batch_size)
            self.image_size = config.get("image_size", self.image_size)
            
            return config
            
        except Exception as e:
            logger.warning(f"Failed to get adaptive config: {e}, using speed mode defaults")
            # Default to speed mode for safety
            return get_speed_mode_config()
    
    async def stream_frames(
        self,
        video_path: str
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """
        Stream video frames asynchronously with timeout protection.
        Process ALL frames regardless of compute device.
        
        Args:
            video_path: Path to the video file
            
        Yields:
            Tuple[int, np.ndarray]: Frame number and frame data
        """
        start_time = time.time()
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            frame_count = 0
            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time > self.processing_timeout:
                    logger.warning(
                        f"Video processing timeout reached after {elapsed_time:.1f}s "
                        f"on {self.device} device ({frame_count} frames processed)"
                    )
                    break
                
                # Use run_in_executor to prevent blocking the event loop
                ret, frame = await asyncio.get_event_loop().run_in_executor(
                    None, cap.read
                )
                
                if not ret:
                    logger.info(f"Completed processing {frame_count} frames in {elapsed_time:.1f}s on {self.device} device")
                    break
                
                yield frame_count, frame
                frame_count += 1
                
                # Small delay to prevent CPU hogging while still processing all frames
                await asyncio.sleep(0)
        
        finally:
            cap.release()
    
    async def stream_frames_batched(
        self,
        video_path: str
    ) -> AsyncGenerator[Tuple[List[int], List[np.ndarray]], None]:
        """
        Stream video frames in batches for RTX 4090 optimization.
        
        Args:
            video_path: Path to the video file
            
        Yields:
            Tuple[List[int], List[np.ndarray]]: Batch of frame numbers and frame data
        """
        # Get adaptive configuration
        config = self._get_adaptive_config(video_path)
        batch_size = config.get("batch_size", self.batch_size)
        image_size = config.get("image_size", self.image_size)
        
        start_time = time.time()
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            frame_numbers = []
            frames = []
            
            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time > self.processing_timeout:
                    logger.warning(
                        f"Video processing timeout reached after {elapsed_time:.1f}s "
                        f"on {self.device} device"
                    )
                    break
                
                # Collect frames for batch processing
                for _ in range(batch_size):
                    ret, frame = await asyncio.get_event_loop().run_in_executor(
                        None, cap.read
                    )
                    
                    if not ret:
                        break
                    
                    # Resize frame for optimal processing
                    if self.device == "cuda":
                        frame = self._resize_frame_for_gpu(frame, image_size)
                    
                    frame_numbers.append(len(frame_numbers))
                    frames.append(frame)
                
                if not frames:
                    break
                
                yield frame_numbers, frames
                
                # Clear batch for next iteration
                frame_numbers = []
                frames = []
                
                # Small delay to prevent CPU hogging
                await asyncio.sleep(0)
        
        finally:
            cap.release()
    
    def _resize_frame_for_gpu(self, frame: np.ndarray, target_size: int = None) -> np.ndarray:
        """Resize frame to optimal size for RTX 4090 processing."""
        if target_size is None:
            target_size = self.image_size
            
        if frame.shape[1] != target_size:  # width
            aspect_ratio = frame.shape[1] / frame.shape[0]
            new_width = target_size
            new_height = int(new_width / aspect_ratio)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return frame
    
    def scale_coordinates(self, coords: List[float], from_size: Tuple[int, int], to_size: Tuple[int, int]) -> List[float]:
        """
        Scale coordinates from model input size to expected output size.
        
        Args:
            coords: [x, y] or [x1, y1, x2, y2] coordinates
            from_size: (width, height) of model input
            to_size: (width, height) of expected output
            
        Returns:
            Scaled coordinates
        """
        if len(coords) == 2:  # Single point [x, y]
            x, y = coords
            scaled_x = (x / from_size[0]) * to_size[0]
            scaled_y = (y / from_size[1]) * to_size[1]
            return [scaled_x, scaled_y]
        elif len(coords) == 4:  # Bounding box [x1, y1, x2, y2]
            x1, y1, x2, y2 = coords
            scaled_x1 = (x1 / from_size[0]) * to_size[0]
            scaled_y1 = (y1 / from_size[1]) * to_size[1]
            scaled_x2 = (x2 / from_size[0]) * to_size[0]
            scaled_y2 = (y2 / from_size[1]) * to_size[1]
            return [scaled_x1, scaled_y1, scaled_x2, scaled_y2]
        else:
            return coords
    
    @staticmethod
    def get_video_info(video_path: str) -> sv.VideoInfo:
        """Get video information using supervision."""
        return sv.VideoInfo.from_video_path(video_path)
    
    @staticmethod
    async def ensure_video_readable(video_path: str, timeout: float = 5.0) -> bool:
        """
        Check if video is readable within timeout period.
        
        Args:
            video_path: Path to video file
            timeout: Maximum time to wait for video check
            
        Returns:
            bool: True if video is readable
        """
        try:
            async def _check_video():
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    return False
                ret, _ = cap.read()
                cap.release()
                return ret
            
            return await asyncio.wait_for(_check_video(), timeout)
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout while checking video readability: {video_path}")
            return False
        except Exception as e:
            logger.error(f"Error checking video readability: {str(e)}")
            return False
    
    def get_processing_config(self, video_path: str) -> Dict[str, Any]:
        """
        Get processing configuration for a specific video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Configuration dictionary
        """
        return self._get_adaptive_config(video_path) 