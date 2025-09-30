import torch.nn as nn
import torch.nn.functional as F

# =====================
# MLP
# =====================

class MLP(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), hidden_size=128, num_classes=10):
        super().__init__()
        c, h, w = input_shape
        self.input_dim = c * h * w
        self.fc1 = nn.Linear(self.input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # aplatissement générique
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ====================
# Logistic regression
# ====================

class LogisticRegression(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        super().__init__()
        c, h, w = input_shape
        self.input_dim = c * h * w
        self.fc = nn.Linear(self.input_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ====================
# CNN
# ====================

class CNN(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        super().__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * (h // 4) * (w // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)