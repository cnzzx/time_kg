import torch
import torch.nn as nn


class FNN_Model(nn.Module):
    """
    This is the pure LSTM model for time series forecasting task.
    """
    def __init__(self, input_size, hidden_size, num_layers, input_step_len, output_step_len):
        super(FNN_Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_step_len = input_step_len
        self.output_step_len = output_step_len
        self.fc1 = nn.Linear(input_step_len, 10)
        self.fc2 = nn.Linear(10, output_step_len)
    
    def forward(self, x):
        x1 = nn.ReLU(self.fc1(x))
        x2 = self.fc2(x1)
        return x2
