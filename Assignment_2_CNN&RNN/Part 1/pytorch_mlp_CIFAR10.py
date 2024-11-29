from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn

class MLP(nn.Module):
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(MLP, self).__init__()
        
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        
        self.n_layers = len(n_hidden)
        
        self.layers = nn.ModuleList()
        
        # Add input layer
        self.layers.append(nn.Linear(n_inputs, n_hidden[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.05))
        self.layers.append(nn.BatchNorm1d(n_hidden[0]))
        
        for i in range(self.n_layers-1):
            self.layers.append(nn.Linear(n_hidden[i], n_hidden[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.05))
            self.layers.append(nn.BatchNorm1d(n_hidden[i+1]))
        
        # Add output layer
        self.layers.append(nn.Linear(n_hidden[-1], n_classes))
        # self.layers.append(nn.Softmax(dim=1))
        
        # self.network = nn.Sequential(*self.layers)
        
        

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        
        out = x.view(x.size(0), -1)
        for layer in self.layers:
            out = layer(out)

        # out = x.view(x.size(0), -1)
        # print(f'Input shape: {out.shape}')
        
        # for i, layer in enumerate(self.layers):
        #     out = layer(out)
        #     print(f'After layer {i + 1} ({layer.__class__.__name__}), shape: {out.shape}')
  
    
        
        return out
