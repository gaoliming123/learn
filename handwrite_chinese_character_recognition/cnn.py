#!/usr/bin/env python
# coding=utf-8

import torch.nn as nn

"""
Module: CnnHwdb
function: convolutional  
"""

class CnnHwdb(nn.Module):

    def __init__(self):
        super(CnnHwdb, self).__init__()

        self.recognize_size = 25
        # the first convolutional
        self.conv1 = nn.Sequential(
           nn.Conv2d(
                in_channels  = 1,                   # input -> 64 * 64 gray image
                out_channels = 32,
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2) # out -> 32 * (32 * 32) matrix
        )

        # the second convolutional
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels  = 32,
                out_channels = 64,
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)  # out -> 64 * (16 * 16) matrix
        )

        # the third convolutional
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels  = 64,
                out_channels = 128,
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2) # out -> 128 * (8 * 8) matrix
        )

        # the forth convolutional
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels  = 128,
                out_channels = 256,
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2) # out -> 256 * (4 * 4) matritx
        )

        # the output linear, what is activation
        self.out = nn.Linear(256 * 4 * 4, self.recognize_size)

    def forward(self, x):
        """
        function: forward update net
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)                # 展开卷积图称(bitch_size, 256 * 4 * 4)
        output = self.out(x)

        return output
