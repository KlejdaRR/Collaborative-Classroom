import torch.nn as nn
import torch.nn.functional as F


class Student(nn.Module):
    def __init__(self, input_channels=3, image_size=32, num_classes=10):
        super(Student, self).__init__()

        # CIFAR-10 images are 32x32 with 3 color channels (RGB)
        # We'll use a deeper architecture suitable for color images

        # Convolutional layers for feature extraction
        # Conv Layer 1: 3 → 32 channels
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm for stable training

        # Conv Layer 2: 32 → 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Conv Layer 3: 64 → 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling layer (shared)
        self.pool = nn.MaxPool2d(2, 2)  # Reduces size by half

        # After 3 pooling layers: 32 → 16 → 8 → 4
        # 128 channels * 4 * 4 = 2048 features
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.3)  # Slightly higher dropout for CIFAR

    def forward(self, x):
        # Input: [batch, 3, 32, 32]

        # Conv Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32 → 16
        x = self.dropout(x)

        # Conv Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16 → 8
        x = self.dropout(x)

        # Conv Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8 → 4
        x = self.dropout(x)

        # Flatten for fully connected layers
        x = x.view(-1, 128 * 4 * 4)  # [batch, 2048]

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Raw logits

        return F.log_softmax(x, dim=1)

    # ARCHITECTURE VISUALIZATION:
    #
    # INPUT: [batch, 3, 32, 32] - Color image
    #     ↓
    # Conv2d(3→32, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
    #     ↓ [batch, 32, 16, 16]
    # Dropout(0.3)
    #     ↓
    # Conv2d(32→64, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
    #     ↓ [batch, 64, 8, 8]
    # Dropout(0.3)
    #     ↓
    # Conv2d(64→128, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
    #     ↓ [batch, 128, 4, 4]
    # Dropout(0.3)
    #     ↓
    # Flatten → [batch, 2048]
    #     ↓
    # Linear(2048→256) + ReLU + Dropout
    #     ↓
    # Linear(256→128) + ReLU + Dropout
    #     ↓
    # Linear(128→10) + LogSoftmax
    #     ↓
    # OUTPUT: 10 log-probabilities (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)