from pathlib import Path
from typing import Dict, Optional
from ultralytics import YOLO
from loguru import logger
import torch
import supervision as sv

from miner.utils.device import get_optimal_device, get_rtx4090_optimal_batch_size, get_rtx4090_optimal_image_size
from scripts.download_models import download_models

# Models downloaded from Hugging Face
REPO_ID = "tmoklc/scorevisionv1"
MODELS = [
    "football-player-detection.pt",  # Detects: players, goalkeepers, referees
    "football-ball-detection.pt",    # Detects: ball
    "football-pitch-detection.pt"    # Detects: pitch keypoints
]

# Class ID mapping (from validator code, but used by miner)
BALL_CLASS_ID = 0        # Ball detection
GOALKEEPER_CLASS_ID = 1  # Goalkeeper detection  
PLAYER_CLASS_ID = 2      # Regular player detection
REFEREE_CLASS_ID = 3     # Referee detection

class ModelManager:
    """Manages the loading and caching of YOLO models optimized for RTX 4090."""
    
    def __init__(self, device: Optional[str] = None):
        self.device = get_optimal_device(device)
        self.models: Dict[str, YOLO] = {}
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # RTX 4090 specific optimizations
        self.batch_size = get_rtx4090_optimal_batch_size()
        self.image_size = get_rtx4090_optimal_image_size()
        
        # Define model paths
        self.model_paths = {
            "player": self.data_dir / "football-player-detection.pt",  # Main detection model
            "pitch": self.data_dir / "football-pitch-detection.pt",    # Pitch keypoints
            "ball": self.data_dir / "football-ball-detection.pt"      # Ball detection
        }
        
        # Check if models exist, download if missing
        self._ensure_models_exist()
        
        # Pre-allocate GPU memory for optimal performance
        if self.device == "cuda":
            self._preallocate_gpu_memory()
    
    def _preallocate_gpu_memory(self):
        """Pre-allocate GPU memory for RTX 4090 to avoid fragmentation."""
        try:
            # Allocate a small tensor to warm up GPU memory
            dummy_tensor = torch.zeros(1, 3, 1920, 1920, device="cuda")
            del dummy_tensor
            torch.cuda.empty_cache()
            logger.info("GPU memory pre-allocated for RTX 4090")
        except Exception as e:
            logger.warning(f"Failed to pre-allocate GPU memory: {e}")
    
    def _ensure_models_exist(self) -> None:
        """Check if required models exist, download if missing."""
        missing_models = [
            name for name, path in self.model_paths.items() 
            if not path.exists()
        ]
        
        if missing_models:
            logger.info(f"Missing models: {', '.join(missing_models)}")
            logger.info("Downloading required models...")
            download_models()
    
    def _optimize_model_for_rtx4090(self, model: YOLO) -> YOLO:
        """Apply RTX 4090 specific optimizations to a model."""
        if self.device == "cuda":
            # Move to GPU
            model = model.to(device=self.device)
            
            # Enable TensorRT for faster inference
            model.export(format="engine", device=self.device, half=True)
            
            # Enable mixed precision
            model.fuse()
            
            # Set optimal parameters
            model.conf = 0.25  # Confidence threshold
            model.iou = 0.45   # NMS IoU threshold
            
            logger.info(f"Model optimized for RTX 4090 with batch_size={self.batch_size}, image_size={self.image_size}")
        
        return model
    
    def load_model(self, model_name: str) -> YOLO:
        """
        Load a model by name, using cache if available.
        
        Args:
            model_name: Name of the model to load ('player', 'pitch', or 'ball')
            
        Returns:
            YOLO: The loaded model
        """
        if model_name in self.models:
            return self.models[model_name]
        
        if model_name not in self.model_paths:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_path = self.model_paths[model_name]
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Please ensure all required models are downloaded."
            )
        
        logger.info(f"Loading {model_name} model from {model_path} to {self.device}")
        model = YOLO(str(model_path))
        
        # Apply RTX 4090 optimizations
        model = self._optimize_model_for_rtx4090(model)
        
        self.models[model_name] = model
        return model
    
    def load_all_models(self) -> None:
        """Load all models into cache."""
        for model_name in self.model_paths.keys():
            self.load_model(model_name)
    
    def get_model(self, model_name: str) -> YOLO:
        """
        Get a model by name, loading it if necessary.
        
        Args:
            model_name: Name of the model to get ('player', 'pitch', or 'ball')
            
        Returns:
            YOLO: The requested model
        """
        return self.load_model(model_name)
    
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for current device."""
        return self.batch_size
    
    def get_optimal_image_size(self) -> int:
        """Get optimal image size for current device."""
        return self.image_size
    
    def clear_cache(self) -> None:
        """Clear the model cache and GPU memory."""
        self.models.clear()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared") 