"""
   kuzu.py
   COMP9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        self.linear = nn.Linear(28 * 28, 10) 

    def forward(self, x):
        # CHANGE CODE HERE
        x = x.view(-1, 28 * 28) 
        x = self.linear(x)       
        return F.log_softmax(x, dim=1)  

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self, hidden_size=100):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.tanh(self.fc1(x))    
        return F.log_softmax(self.fc2(x), dim=1)

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7) 
        x = self.fc1(x)
        return F.log_softmax(x, dim=1) # CHANGE CODE HERE
