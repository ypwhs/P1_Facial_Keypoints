## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import numpy as np
import torchvision.models as models

class Net(nn.Module):

    def __init__(self, input_shape=(3, 224, 224)):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        def conv_block(indim, kernels):
        	return (
                nn.Conv2d(indim, kernels, 3), 
                nn.BatchNorm2d(kernels), 
                nn.ELU(), 
                nn.Conv2d(kernels, kernels, 3), 
                nn.BatchNorm2d(kernels), 
                nn.ELU(), 
                nn.MaxPool2d((2,2)),
                )

        self.features = nn.Sequential(
            *conv_block(3, 64), 
            *conv_block(64, 128), 
            *conv_block(128, 128), 
            *conv_block(128, 128), 
            *conv_block(128, 128), 
        )
        
        self.feature_dim = self.get_features_dim(input_shape)
        
        self.hidden_units = 1000
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_units), 
            nn.BatchNorm1d(self.hidden_units), 
            nn.ELU(), 
            nn.Dropout(p=0.5), 

            nn.Linear(self.hidden_units, self.hidden_units), 
            nn.BatchNorm1d(self.hidden_units), 
            nn.ELU(), 
            nn.Dropout(p=0.5), 

            nn.Linear(self.hidden_units, 136), 
        )
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
    def get_features_dim(self, shape):
        features = self.features(Variable(torch.ones(1, *shape)))
        return int(np.prod(features.size()[1:]))
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.features(x)
        x = x.view(-1, self.feature_dim)
        x = self.regressor(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
