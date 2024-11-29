from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class VanillaRNN(nn.Module):

    def __init__(self, input_length, input_dim, hidden_dim, output_dim):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        
        self.input_length = input_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.i2h = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()
        

    def forward(self, x):
        # Implementation here ...
        
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        outputs = []

        # print(f"x.shape: {x.shape}")
        # print(f"h_t.shape: {h_t.shape}")

        # Iterate through each time step in the sequence
        for t in range(self.input_length):
            # Concatenate input at time step t with the previous hidden state
            combined = torch.cat((x[:, t, :], h_t), dim=1)
            
            # print(f"combined.shape: {combined.shape}")
            
            # Update the hidden state
            h_t = self.tanh(self.i2h(combined))
            
            outputs.append(self.h2o(h_t))
        
        return outputs
        
    # add more methods here if needed
