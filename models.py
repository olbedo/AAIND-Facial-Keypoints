## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # 1st convolutional layer (kernel: 5x5x32)
        self.conv1 = nn.Conv2d(1, 16, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # 2nd convolutional layer (kernel: 3x3x64)
        self.conv2 = nn.Conv2d(16, 32, 3)
        # 3rd convolutional layer (kernel: 2x2x128)
        self.conv3 = nn.Conv2d(32, 64, 2)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # 1st fully connected layer
        self.fc1 = nn.Linear(64*26*26, 1000)
        # 2nd fully connected layer
        self.fc2 = nn.Linear(1000, 136)
        # dropout layers
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        # 1st convolutional layer with activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # dropout layer
        x = self.dropout1(x)
        # 2nd convolutional layer with activation function and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        # dropout layer
        x = self.dropout2(x)
        # 3rd convolutional layer with activation function and max pooling
        x = self.pool(F.relu(self.conv3(x)))
        # dropout layer
        x = self.dropout3(x)
        # flatten image input
        x = x.view(-1, 64*26*26)
        # 1st fully connected layer with activation function
        x = F.relu(self.fc1(x))
        # dropout layer
        x = self.dropout4(x)
        # 2nd fully connected layer (without activation function)
        x = self.fc2(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x


class NaimishNet(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # 1st convolutional layer (kernel: 5x5x32)
        self.conv1 = nn.Conv2d(1, 32, 5)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # 2nd convolutional layer (kernel: 3x3x64)
        self.conv2 = nn.Conv2d(32, 64, 3)
        # 3rd convolutional layer (kernel: 2x2x128)
        self.conv3 = nn.Conv2d(64, 128, 2)
        # 4th convolutional layer (kernel: 1x1x256)
        self.conv4 = nn.Conv2d(128, 256, 1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # 1st fully connected layer
        self.fc1 = nn.Linear(256 * 13 * 13, 6400)
        # 2nd fully connected layer
        self.fc2 = nn.Linear(6400, 1000)
        # 3rd fully connected layer
        self.fc3 = nn.Linear(1000, 136)
        # dropout layers
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout6 = nn.Dropout(0.6)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        # 1st convolutional layer with activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # dropout layer
        x = self.dropout1(x)
        # 2nd convolutional layer with activation function and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        # dropout layer
        x = self.dropout2(x)
        # 3rd convolutional layer with activation function and max pooling
        x = self.pool(F.relu(self.conv3(x)))
        # dropout layer
        x = self.dropout3(x)
        # 4th convolutional layer with activation function and max pooling
        x = self.pool(F.relu(self.conv4(x)))
        # flatten image input
        x = x.view(-1, 256 * 13 * 13)
        # dropout layer
        x = self.dropout4(x)
        # 1st fully connected layer with activation function
        x = F.relu(self.fc1(x))
        # dropout layer
        x = self.dropout5(x)
        # 2nd fully connected layer with activation function
        x = F.relu(self.fc2(x))
        # dropout layer
        x = self.dropout6(x)
        # 3rd fully connected layer (without activation function)
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
