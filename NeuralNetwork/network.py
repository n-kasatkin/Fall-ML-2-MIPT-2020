import torch
import torch.nn as nn


class NNClassifier(nn.Module):
    def __init__(self, input_size=100, num_classes=30):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.PReLU(),
            nn.BatchNorm1d(input_size),
            nn.Dropout(0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.PReLU(),
            nn.BatchNorm1d(input_size)
        )
        self.out = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x