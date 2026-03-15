"""
CNN MODEL FOR MNIST DIGIT CLASSIFICATION

CNN = Convolutional Neural Network

CNNs are specialized neural networks designed for images.

Instead of looking at the entire image at once,
CNN scans small parts of the image using filters.

These filters learn patterns like:

edges
curves
shapes
digit structures

MNIST images:
28 x 28 pixels
grayscale (1 channel)
"""

# ---------------------------------------------------------
# Import required PyTorch libraries
# ---------------------------------------------------------

import torch

# nn contains neural network building blocks
import torch.nn as nn

# functional API provides activation functions like ReLU
import torch.nn.functional as F


class CNN(nn.Module):

    """
    CNN architecture used for MNIST classification.

    Structure:

    Input Image (1x28x28)
          │
      Conv Layer
          │
      ReLU
          │
      Max Pool
          │
      Conv Layer
          │
      ReLU
          │
      Max Pool
          │
      Flatten
          │
      Fully Connected
          │
      Output (10 classes)
    """

    def __init__(self):

        # initialize parent class
        super(CNN, self).__init__()

        # -------------------------------------------------
        # First Convolution Layer
        # -------------------------------------------------
        #
        # Parameters:
        #
        # in_channels = 1
        # because MNIST images are grayscale
        #
        # out_channels = 32
        # meaning we learn 32 filters
        #
        # kernel_size = 3
        # filter size = 3x3
        #
        # This layer detects simple patterns like edges
        #
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3
        )

        # -------------------------------------------------
        # Second Convolution Layer
        # -------------------------------------------------
        #
        # This layer takes the 32 feature maps produced
        # by conv1 and produces 64 new feature maps.
        #
        # Deeper layers learn more complex patterns.
        #
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )

        # -------------------------------------------------
        # Fully Connected Layer
        # -------------------------------------------------
        #
        # After convolutions and pooling,
        # the feature maps become:
        #
        # 64 channels
        # 5 x 5 spatial size
        #
        # total features = 64 * 5 * 5 = 1600
        #
        self.fc1 = nn.Linear(1600, 128)

        # -------------------------------------------------
        # Output Layer
        # -------------------------------------------------
        #
        # MNIST has 10 classes:
        #
        # digits 0 to 9
        #
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        """
        Forward pass describes how data flows through the network.
        """

        # -------------------------------------------------
        # First convolution
        # -------------------------------------------------
        #
        # Input shape:
        #
        # (batch, 1, 28, 28)
        #
        x = self.conv1(x)

        # Apply ReLU activation
        #
        # ReLU introduces non-linearity
        # allowing the network to learn complex patterns
        #
        x = F.relu(x)

        # -------------------------------------------------
        # First Max Pooling
        # -------------------------------------------------
        #
        # Pooling reduces spatial dimensions.
        #
        # This helps:
        #
        # reduce computation
        # reduce overfitting
        # focus on important features
        #
        # 26x26 → 13x13
        #
        x = F.max_pool2d(x, 2)

        # -------------------------------------------------
        # Second convolution
        # -------------------------------------------------
        #
        # 32 feature maps → 64 feature maps
        #
        x = self.conv2(x)

        x = F.relu(x)

        # -------------------------------------------------
        # Second Max Pool
        # -------------------------------------------------
        #
        # 11x11 → 5x5
        #
        x = F.max_pool2d(x, 2)

        # -------------------------------------------------
        # Flatten tensor
        # -------------------------------------------------
        #
        # CNN output is a 3D tensor:
        #
        # (batch, channels, height, width)
        #
        # Fully connected layers expect a vector.
        #
        # So we flatten:
        #
        # 64 x 5 x 5 = 1600
        #
        x = torch.flatten(x, 1)

        # -------------------------------------------------
        # First Fully Connected Layer
        # -------------------------------------------------
        #
        # This layer combines extracted features
        # to make a decision.
        #
        x = F.relu(self.fc1(x))

        # -------------------------------------------------
        # Output Layer
        # -------------------------------------------------
        #
        # Produces 10 values (logits)
        # representing probabilities for digits 0-9
        #
        x = self.fc2(x)

        return x