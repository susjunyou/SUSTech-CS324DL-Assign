from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.i2h = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        
        self.h2o = nn.Linear(hidden_dim, output_dim)
    
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Implementation here ...
        batch_size = x.size(0)
        h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        outputs = []
        
        for t in range(self.seq_length):
            combined = torch.cat((x[:, t, :], h_t), dim=1)
            gates = self.i2h(combined)
            i_t, f_t, g_t, o_t = gates.chunk(4, dim=1)
            
            i_t = self.sigmoid(i_t)
            f_t = self.sigmoid(f_t)
            g_t = self.tanh(g_t)
            o_t = self.sigmoid(o_t)
            
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * self.tanh(c_t)
            
            outputs.append(self.h2o(h_t))

        return outputs[-1]
        
    # add more methods here if needed