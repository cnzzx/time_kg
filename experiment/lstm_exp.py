import os
import sys
sys.path.append(os.path.abspath('..'))

import torch
import data.load as load
from net.lstm import LSTM_Model


# hyper-parameters
input_size = 1
hidden_size = 1
num_layers = 2
input_step_len = 10
output_step_len = 10
eval_ratio = 0.2  # 20 % for evaluation, 80 % for training
valid_ratio = 0.2  # in the training set, use 20 % for validation
n_epochs = 100
batch_size = 64
dataset_path = '../data/dataset/nasdaq100.csv'
dataset_name = 'nasdaq100'

# prepare dataset
nasdaq100 = load.data_filter(load.load_data(dataset_path, dataset_name), 7)
n_samples = len(nasdaq100)
values = []
for item in nasdaq100:
    values.append(item[1])  # get the value only
eval_values = values[int(n_samples * (1-eval_ratio)):]
valid_values = values[int(n_samples * (1-eval_ratio) * (1-valid_ratio)):int(n_samples * (1-eval_ratio))]
train_values = values[:int(n_samples * (1-eval_ratio) * (1-valid_ratio))]

train_input_data, train_output_data = load.make_dataset(train_values, input_step_len, output_step_len)
valid_input_data, valid_output_data = load.make_dataset(valid_values, input_step_len, output_step_len)
eval_input_data, eval_output_data = load.make_dataset(eval_values, input_step_len, output_step_len)
train_input_tensor = torch.transpose(torch.tensor(train_input_data).unsqueeze(2), 0, 1)
train_output_tensor = torch.transpose(torch.tensor(train_output_data).unsqueeze(2), 0, 1)
valid_input_tensor = torch.transpose(torch.tensor(valid_input_data).unsqueeze(2), 0, 1)
valid_output_tensor = torch.transpose(torch.tensor(valid_output_data).unsqueeze(2), 0, 1)
eval_input_tensor = torch.transpose(torch.tensor(eval_input_data).unsqueeze(2), 0, 1)
eval_output_tensor = torch.transpose(torch.tensor(eval_output_data).unsqueeze(2), 0, 1)
n_train = train_input_tensor.size()[1]
n_valid = valid_input_tensor.size()[1]
n_eval = eval_input_tensor.size()[1]

# prepare model
model = LSTM_Model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, input_step_len=input_step_len, output_step_len=output_step_len)

# training
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()
n_batch = (n_train-1) // batch_size + 1
for epoch_idx in range(n_epochs):
    for batch_idx in range(n_batch):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx+1)*batch_size, n_train)
        x = train_input_tensor[:, start_idx:end_idx, :]
        x.requires_grad = True
        y = train_output_tensor[:, start_idx:end_idx, :]
        out = model(x)
        
        optimizer.zero_grad()
        loss = loss_func(y, out)
        loss.backward()
        optimizer.step()

        print('train_loss: ', x.grad)
        break
    break

# evaluation

