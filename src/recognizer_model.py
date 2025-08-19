import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from model_config import BaseConfig

class RecognizerModel(nn.Module, ABC):
    """
    Abstract Base Class for all CAPTCHA recognizer models.

    This class establishes a common interface for any model intended for
    CAPTCHA recognition within this project. It ensures that every model:
    1. Inherits from `torch.nn.Module`, making it a valid PyTorch model.
    2. Accepts a `BaseConfig` object during initialization, providing
       essential parameters like image dimensions and character set length.
    3. Implements a `forward` method that takes an image tensor and returns an
       output tensor.
    """
    def __init__(self, model_config: BaseConfig):
        """
        Initializes the base model.

        Args:
            model_config (BaseConfig): The base configuration object containing
                                       shared model and data parameters.
        """
        super().__init__()
        self.model_config = model_config
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        This method must be implemented by all subclasses.

        Args:
            x (torch.Tensor): The input tensor, typically a batch of images
                              with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output tensor from the model. For this project,
                          it should be the raw logits with shape
                          (batch_size, max_captcha * char_set_len).
        """
        pass
    