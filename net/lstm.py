from pyexpat import model
import torch
import torch.nn as nn


class LSTM_Model(nn.Module):
    """
    This is the pure LSTM model for time series forecasting task.
    """
    def __init__(self, input_size, hidden_size, num_layers, input_step_len, output_step_len):
        super(LSTM_Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_step_len = input_step_len
        self.output_step_len = output_step_len
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(input_step_len * hidden_size, output_step_len)
    
    def forward(self, x):
        out, (h, c) = self.lstm(x)
        h_T = torch.transpose(h, 0, 2)
        r_T = self.fc(h_T)
        r = torch.transpose(r_T, 0, 2)
        return r
