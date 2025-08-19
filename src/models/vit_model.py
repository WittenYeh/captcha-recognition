import torch
import torch.nn as nn
from torchvision.models import vit_b_16
# Add this import
import torchvision.transforms.functional as F
from model_config import BaseConfig
from recognizer_model import RecognizerModel
from typing_extensions import override

class ViTModel(RecognizerModel):
    def __init__(self, model_config: BaseConfig):
        super().__init__(model_config)
        
        # Define the target image size for ViT
        self.image_size = 224

        # Load the pretrained ViT model
        self.vit = vit_b_16(weights=None)

        # --- Adapt the first convolutional layer for 1-channel (grayscale) input ---
        
        # Get the original weights
        original_conv_stem_weights = self.vit.conv_proj.weight.data
        
        # Create a new conv layer with 1 input channel
        self.vit.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        
        # Average the weights of the original 3 channels and assign to the new layer
        self.vit.conv_proj.weight.data = torch.mean(original_conv_stem_weights, dim=1, keepdim=True)

        # --- Replace the final fully connected layer (head) ---
        num_ftrs = self.vit.heads.head.in_features
        output_size = model_config.max_captcha * model_config.char_set_len
        self.vit.heads.head = nn.Linear(num_ftrs, output_size)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Resize the image to the size expected by ViT ---
        # Use bilinear interpolation for resizing
        x = F.resize(x, [self.image_size, self.image_size], antialias=True)
        
        # Now the image has the correct size and can be processed by the model
        return self.vit(x)