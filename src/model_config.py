
import os
import torch
from dataclasses import dataclass, field
from enum import Enum, auto
import time

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
UPCASE_ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

class ModelType(Enum):
    CNN = auto()
    RESNET = auto()
    VGG = auto()
    VIT = auto()

@dataclass
class BaseConfig:
    # --- captcha configuration ---
    char_set: list[str] = field(default_factory=lambda: UPCASE_ALPHABET)
    max_captcha: int = 4
    image_height: int = 60
    image_width: int = 160

    @property
    def char_set_len(self) -> int:
        return len(self.char_set)

    # --- model and device configuration ---
    model_type: ModelType = ModelType.CNN
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids: list[int] = field(default_factory=lambda: [0])

@dataclass
class TrainingConfig(BaseConfig):
    # --- dataset and dataloader configuration ---
    train_dataset_size: int = 299985
    test_dataset_size: int = 29996
    val_dataset_size: int = 29996
    
    # --- training configuration ---
    epochs: int = 60
    batch_size: int = 1024
    learning_rate: float = 1e-4
    
    # --- path config ---
    train_dataset_path: str = '../dataset/train/'
    test_dataset_path: str = '../dataset/test/'
    val_dataset_path: str = '../dataset/val/'
    
    # --- result and checkpoint paths ---
    result_path: str = '../result/'
    
    @property
    def model_name(self) -> str:
        # Generate a unique model name with timestamp
        return f'{self.model_type.name}'

    @property
    def model_save_path(self) -> str:
        # Path to save the best model weights
        return os.path.join('../model_weight/', f'{self.model_name}_best_model.pth')

    @property
    def checkpoint_dir(self) -> str:
        # Directory to save checkpoints
        return os.path.join('../model_weight/checkpoints/', self.model_type.name)

@dataclass
class InferenceConfig(BaseConfig):
    # --- path config ---
    model_save_path: str = '../model_weight/CNN.pth'
    test_dataset_path: str = '../dataset/test/'
    
    # --- inference configuration ---
    batch_size: int = 1024 # Batch size for testing
    regenerate_cache: bool = False
