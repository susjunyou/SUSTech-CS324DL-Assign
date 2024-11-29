from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

class CNN(nn.Module):

    def __init__(self, n_channels, n_classes):
        """
        Initializes CNN object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
        """
        super(CNN, self).__init__()
        
        # First block
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Second block
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Third block
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Fourth block
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Fifth block
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Fully connected layer
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        """
        Performs forward pass of the input.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        # First block
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        
        # Third block
        x = self.pool3(self.relu4(self.bn4(self.conv4(self.relu3(self.bn3(self.conv3(x)))))))
        
        # Fourth block
        x = self.pool4(self.relu6(self.bn6(self.conv6(self.relu5(self.bn5(self.conv5(x)))))))
        
        # Fifth block
        x = self.pool5(self.relu8(self.bn8(self.conv8(self.relu7(self.bn7(self.conv7(x)))))))
        
        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)  # Flatten the output
        out = self.fc(x)
        return out
    
    # def forward(self, x):
    #   """
    #   Performs forward pass of the input.
      
    #   Args:
    #     x: input to the network
    #   Returns:
    #     out: outputs of the network
    #   """
      
    #   print("Input shape:", x.shape)
      
    #   # First block
    #   x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
    #   print("After first block:", x.shape)
      
    #   # Second block
    #   x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
    #   print("After second block:", x.shape)
      
    #   # Third block
    #   x = self.pool3(self.relu4(self.bn4(self.conv4(self.relu3(self.bn3(self.conv3(x)))))))
    #   print("After third block:", x.shape)
      
    #   # Fourth block
    #   x = self.pool4(self.relu6(self.bn6(self.conv6(self.relu5(self.bn5(self.conv5(x)))))))
    #   print("After fourth block:", x.shape)
      
    #   # Fifth block
    #   x = self.pool5(self.relu8(self.bn8(self.conv8(self.relu7(self.bn7(self.conv7(x)))))))
    #   print("After fifth block:", x.shape)
      
    #   # Flatten and fully connected layer
    #   x = x.view(x.size(0), -1)  # Flatten the output
    #   print("After flatten:", x.shape)
    #   out = self.fc(x)
    #   print("After fully connected layer:", out.shape)
      
    #   return out