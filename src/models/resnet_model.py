import torch.nn as nn
import torch
from torchvision.models import resnet18
from model_config import BaseConfig
from recognizer_model import RecognizerModel
from typing_extensions import override

class ResNetModel(RecognizerModel):
    def __init__(self, model_config: BaseConfig):
        super().__init__(model_config)
        
        # Load the pretrained ResNet18 model
        self.resnet = resnet18(weights=None)
        
        # --- Adapt the first convolutional layer for 1-channel (grayscale) input ---
        
        # Get the original weights
        original_conv1_weights = self.resnet.conv1.weight.data
        
        # Create a new conv layer with 1 input channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Average the weights of the original 3 channels and assign to the new layer
        self.resnet.conv1.weight.data = torch.mean(original_conv1_weights, dim=1, keepdim=True)
        
        # --- Replace the final fully connected layer ---
        num_ftrs = self.resnet.fc.in_features
        output_size = model_config.max_captcha * model_config.char_set_len
        self.resnet.fc = nn.Linear(num_ftrs, output_size)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
    