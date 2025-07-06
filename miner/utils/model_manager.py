from pathlib import Path
from typing import Dict, Optional
from ultralytics import YOLO
from loguru import logger
import torch
import supervision as sv
import os
import numpy as np
import cv2

from miner.utils.device import get_optimal_device, get_rtx4090_optimal_batch_size, get_rtx4090_optimal_image_size

# Models downloaded from Hugging Face
REPO_ID = "tmoklc/scorevisionv1"
MODELS = [
    "football-player-detection.pt",  # Detects: players, goalkeepers, referees, BALLS
    "football-pitch-detection.pt"    # Detects: pitch keypoints
]

# Class ID mapping (from validator code, but used by miner)
BALL_CLASS_ID = 0        # Ball detection (from player model)
GOALKEEPER_CLASS_ID = 1  # Goalkeeper detection  
PLAYER_CLASS_ID = 2      # Regular player detection
REFEREE_CLASS_ID = 3     # Referee detection

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

class ModelManager:
    """Manages the loading and caching of YOLO models optimized for RTX 4090, with engine reuse and flexible input size."""
    
    def __init__(self, device: Optional[str] = None, input_size=(1280, 704)):
        self.device = get_optimal_device(device)
        self.models: Dict[str, YOLO] = {}
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.input_size = input_size  # (width, height) - 1280x704 is divisible by 32
        
        # Check for skip TensorRT flag
        self.skip_tensorrt = os.getenv('SKIP_TENSORRT', 'false').lower() == 'true'
        if self.skip_tensorrt:
            logger.info("SKIP_TENSORRT enabled - will use PyTorch models directly")
        
        # RTX 4090 specific optimizations
        self.batch_size = get_rtx4090_optimal_batch_size()
        self.image_size = get_rtx4090_optimal_image_size()
        
        # Define model paths
        self.model_paths = {
            "player": self.data_dir / "football-player-detection.pt",  # Main detection model (includes ball detection)
            "pitch": self.data_dir / "football-pitch-detection.pt",    # Pitch keypoints
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
            # Import here to avoid circular import
            from scripts.download_models import download_models
            download_models()
    
    def _optimize_model_for_rtx4090(self, model: YOLO) -> YOLO:
        """Apply RTX 4090 specific optimizations to a model."""
        if self.device == "cuda":
            # Move to GPU
            model = model.to(device=self.device)
            
            # Check for speed mode environment variable
            speed_mode = os.getenv("FORCE_SPEED_MODE", "false").lower() == "true"
            
            if speed_mode:
                # Speed mode optimizations
                model.conf = 0.15  # Lower confidence threshold for speed
                model.iou = 0.35   # Lower NMS threshold for speed
                logger.info(f"Model optimized for RTX 4090 SPEED MODE with batch_size={self.batch_size}, image_size={self.image_size}")
            else:
                # Standard optimizations
                model.conf = 0.25  # Standard confidence threshold
                model.iou = 0.45   # Standard NMS threshold
                logger.info(f"Model optimized for RTX 4090 with batch_size={self.batch_size}, image_size={self.image_size}")
            
            # Enable mixed precision
            model.fuse()
        
        return model
    
    def _get_engine_path(self, model_path, input_size):
        # Engine file name encodes input size for uniqueness
        w, h = input_size
        return Path(str(model_path).replace('.pt', f'_{w}x{h}.engine'))

    def load_model(self, model_name: str) -> YOLO:
        """
        Load a model by name, using cache if available. Reuse TensorRT engine if available.
        Args:
            model_name: Name of the model to load ('player' or 'pitch')
        Returns:
            YOLO: The loaded model
        """
        if model_name in self.models:
            return self.models[model_name]
        
        if model_name not in self.model_paths:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_path = self.model_paths[model_name]
        engine_path = self._get_engine_path(model_path, self.input_size)
        
        # PRIORITY: Always try to load existing engine first
        if engine_path.exists():
            logger.info(f"Loading existing TensorRT engine for {model_name} from {engine_path}")
            try:
                model = YOLO(str(engine_path))
                # Apply RTX 4090 optimizations
                model = self._optimize_model_for_rtx4090(model)
                self.models[model_name] = model
                return model
            except Exception as e:
                logger.warning(f"Failed to load engine {engine_path}: {e}, will export new one")
        
        # Check for any existing engine files for this model
        model_stem = model_path.stem
        possible_engines = list(self.data_dir.glob(f"{model_stem}*.engine"))
        
        if possible_engines:
            # Use the first available engine (different input size is better than recompiling)
            existing_engine = possible_engines[0]
            logger.info(f"Found existing engine {existing_engine} for {model_name}, using it")
            try:
                model = YOLO(str(existing_engine))
                model = self._optimize_model_for_rtx4090(model)
                self.models[model_name] = model
                return model
            except Exception as e:
                logger.warning(f"Failed to load existing engine {existing_engine}: {e}")
        
        # Only export if no engines exist
        logger.info(f"Loading {model_name} model from {model_path} to {self.device}")
        model = YOLO(str(model_path))
        
        # Skip TensorRT if flag is enabled
        if self.skip_tensorrt:
            logger.info(f"Skipping TensorRT compilation for {model_name}, using PyTorch model directly")
        # Export to engine if on CUDA and no engines exist
        elif self.device == "cuda" and not possible_engines:
            logger.info(f"Exporting {model_name} to TensorRT engine for input size {self.input_size}")
            try:
                model.export(format="engine", imgsz=self.input_size, device=self.device, half=True)
                
                # YOLO saves the engine file in the same directory as the model
                # Check multiple possible locations
                possible_engine_paths = [
                    Path("engine.engine"),  # Current directory
                    model_path.parent / "engine.engine",  # Model directory
                    self.data_dir / "engine.engine",  # Data directory
                    # YOLO actually saves with model name prefix
                    model_path.parent / f"{model_path.stem}.engine",  # Model name + .engine
                    self.data_dir / f"{model_path.stem}.engine",  # Data dir + model name + .engine
                ]
                
                engine_file = None
                for path in possible_engine_paths:
                    if path.exists():
                        engine_file = path
                        break
                
                if engine_file:
                    # Rename to our custom path with input size
                    os.rename(str(engine_file), str(engine_path))
                    logger.info(f"TensorRT engine saved to {engine_path}")
                    # Reload as engine
                    model = YOLO(str(engine_path))
                else:
                    logger.warning(f"Engine export completed but file not found in any expected location")
                    logger.warning(f"Checked paths: {[str(p) for p in possible_engine_paths]}")
                    # List what's actually in the data directory
                    logger.info(f"Files in data directory: {list(self.data_dir.glob('*.engine'))}")
            except Exception as e:
                logger.error(f"Failed to export engine for {model_name}: {e}")
        
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

    # Utility: Pad frame to square for square-only models
    @staticmethod
    def pad_to_square(frame, size=1280):
        h, w = frame.shape[:2]
        top = (size - h) // 2
        bottom = size - h - top
        left = (size - w) // 2
        right = size - w - left
        return cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # Utility: Crop/correct output from square to 1280x720
    @staticmethod
    def crop_coords_from_square(coords, orig_w=1280, orig_h=720, padded_size=1280):
        pad_y = (padded_size - orig_h) // 2
        pad_x = (padded_size - orig_w) // 2
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            return [coords[0] - pad_x, coords[1] - pad_y]
        elif isinstance(coords, (list, tuple)) and len(coords) == 4:
            return [
                coords[0] - pad_x, coords[1] - pad_y,
                coords[2] - pad_x, coords[3] - pad_y
            ]
        elif isinstance(coords, (list, tuple)):
            return [ModelManager.crop_coords_from_square(c, orig_w, orig_h, padded_size) for c in coords]
        else:
            raise ValueError("Unsupported coordinate format") 