#!/usr/bin/env python3
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
from loguru import logger
from ultralytics import YOLO
from typing import Optional

# All models in a single repository
REPO_ID = "tmoklc/scorevisionv1"
MODELS = [
    "football-player-detection.pt",
    "football-ball-detection.pt",
    "football-pitch-detection.pt"
]

# Enhanced RTX 4090 configuration
ENHANCED_RTX4090_CONFIG = {
    # Quality mode settings
    "quality_mode": {
        "player_model": "yolov8x.pt",  # Highest quality
        "ball_model": "yolov8n.pt",    # Fast ball detection
        "pitch_model": "football-pitch-detection.pt",
        "player_image_size": 1920,
        "ball_image_size": 640,
        "confidence_threshold": 0.3,
        "enable_post_processing": True
    },
    
    # Speed mode settings
    "speed_mode": {
        "player_model": "football-player-detection.pt",
        "ball_model": "yolov8n.pt",
        "pitch_model": "football-pitch-detection.pt",
        "player_image_size": 1280,
        "ball_image_size": 640,
        "confidence_threshold": 0.25,
        "enable_post_processing": False
    },
    
    # RTX 4090 optimizations
    "rtx4090_optimizations": {
        "batch_size": 12,  # Increased for RTX 4090
        "max_image_size": 1920,
        "enable_tensorrt": True,
        "mixed_precision": True,
        "memory_fraction": 0.95
    }
}

# Enhanced ModelManager for better quality and speed
class EnhancedModelManager(ModelManager):
    """Enhanced ModelManager with fast ball detection and quality optimizations."""
    
    def __init__(self, device: Optional[str] = None):
        super().__init__(device)
        self.fast_ball_model = None
        self.quality_models = {}
        
    def load_fast_ball_model(self) -> YOLO:
        """Load YOLOv8n for fast ball detection."""
        if self.fast_ball_model is not None:
            return self.fast_ball_model
            
        fast_ball_model_path = self.data_dir / "yolov8n.pt"
        
        if not fast_ball_model_path.exists():
            logger.info("Downloading fast ball detection model (YOLOv8n)...")
            model = YOLO("yolov8n.pt")
            model.save(str(fast_ball_model_path))
        else:
            model = YOLO(str(fast_ball_model_path))
        
        # RTX 4090 optimizations
        if self.device == "cuda":
            model.to(device=self.device)
            model.fuse()
            model.conf = 0.25  # Lower confidence for ball detection
            model.iou = 0.45
        
        self.fast_ball_model = model
        return model
    
    def load_quality_model(self, model_name: str) -> YOLO:
        """Load high-quality model for specific tasks."""
        if model_name in self.quality_models:
            return self.quality_models[model_name]
        
        # Use larger models for quality
        quality_model_path = self.data_dir / f"{model_name}-quality.pt"
        
        if model_name == "player" and not quality_model_path.exists():
            # Use YOLOv8x for highest quality player detection
            logger.info("Loading YOLOv8x for high-quality player detection...")
            model = YOLO("yolov8x.pt")
            model.save(str(quality_model_path))
        else:
            model = YOLO(str(quality_model_path))
        
        # Quality optimizations
        if self.device == "cuda":
            model.to(device=self.device)
            model.conf = 0.3  # Higher confidence for quality
            model.iou = 0.5
        
        self.quality_models[model_name] = model
        return model

def download_models():
    """Download required models from Hugging Face."""
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    success = True
    for model_name in MODELS:
        model_path = data_dir / model_name
        if not model_path.exists():
            logger.info(f"Downloading {model_name} from Hugging Face ({REPO_ID})...")
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=model_name,
                    local_dir=data_dir
                )
                logger.info(f"Successfully downloaded {model_name}")
            except (RepositoryNotFoundError, RevisionNotFoundError) as e:
                logger.error(f"Repository or file not found for {model_name}: {str(e)}")
                success = False
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {str(e)}")
                success = False
        else:
            logger.info(f"{model_name} already exists in {model_path}, skipping download")
    
    if not success:
        logger.error("Some models failed to download. Please check the errors above.")
        exit(1)
    else:
        logger.info("All models downloaded successfully!")

if __name__ == "__main__":
    download_models() 