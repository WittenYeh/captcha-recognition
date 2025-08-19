import torch
import torch.nn as nn
from model_config import BaseConfig
from typing_extensions import override
from recognizer_model import RecognizerModel

# CNN Model
class CNNModel(RecognizerModel):
    def __init__(self, model_config: BaseConfig):
        super().__init__(model_config)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear((model_config.image_height // 8) * (model_config.image_width // 8) * 64, 1024),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.rfc = nn.Sequential(
            nn.Linear(1024, model_config.max_captcha * model_config.char_set_len)
        )
    
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.rfc(x)
        return x
    