import asyncio
import time
from typing import AsyncGenerator, Optional, Tuple, List
import cv2
import numpy as np
import supervision as sv
from loguru import logger
import torch

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
    ):
        self.device = device
        self.batch_size = batch_size
        self.image_size = image_size
        
        # Set timeout based on device
        if device == "cuda":
            self.processing_timeout = cuda_timeout
        elif device == "mps":
            self.processing_timeout = mps_timeout
        else:  # cpu or any other device
            self.processing_timeout = cpu_timeout
            
        logger.info(f"Video processor initialized with {device} device, timeout: {self.processing_timeout:.1f}s")
        logger.info(f"RTX 4090 optimizations: batch_size={batch_size}, image_size={image_size}")
    
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
                for _ in range(self.batch_size):
                    ret, frame = await asyncio.get_event_loop().run_in_executor(
                        None, cap.read
                    )
                    
                    if not ret:
                        break
                    
                    # Resize frame for optimal processing
                    if self.device == "cuda":
                        frame = self._resize_frame_for_gpu(frame)
                    
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
    
    def _resize_frame_for_gpu(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to optimal size for RTX 4090 processing."""
        if frame.shape[1] != self.image_size:  # width
            aspect_ratio = frame.shape[1] / frame.shape[0]
            new_width = self.image_size
            new_height = int(new_width / aspect_ratio)
            frame = cv2.resize(frame, (new_width, new_height))
        return frame
    
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