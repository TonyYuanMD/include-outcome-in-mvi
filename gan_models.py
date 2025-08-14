import torch.nn as nn
import torch

class Generator(nn.Module):
    """
    Generator network for GAIN-like GAN.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, input_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Linear output for continuous data

class Discriminator(nn.Module):
    """
    Discriminator network for GAIN-like GAN.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, input_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

# Documentation
"""
Classes:
- Generator: Feedforward NN for imputing missing values.
- Discriminator: Feedforward NN for distinguishing real vs. imputed values.
"""