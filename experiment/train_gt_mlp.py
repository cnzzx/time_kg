"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math

from metrics import MAE

def train_epoch(model, optimizer, device, data_loader, event_graph, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    gpu_mem = 0
    event_graph = event_graph.to(device)
    for iter, (input_seq, output_seq, start_bucket) in enumerate(data_loader):
        start_bucket = start_bucket.item()
        input_seq_len = input_seq.size()[1]
        output_seq_len = output_seq.size()[1]
        end_bucket = start_bucket + (input_seq_len + output_seq_len) - 1
        input_seq = input_seq[0].to(device)
        output_seq = output_seq[0].to(device)
        embeddings = event_graph.ndata['feat'].to(device)
        similarity = event_graph.edata['feat'].to(device)
        optimizer.zero_grad()
        try:
            '''
            ???
            '''
            lap_pos_enc = event_graph.ndata['lap_pos_enc'].to(device)
            sign_flip = torch.rand(lap_pos_enc.size(0)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0
            sign_flip[sign_flip<0.5] = -1.0
            lap_pos_enc = lap_pos_enc * sign_flip.unsqueeze(1)
        except:
            lap_pos_enc = None
            
        try:
            wl_pos_enc = event_graph.ndata['wl_pos_enc'].to(device)
        except:
            wl_pos_enc = None

        forecast = model.forward(device, event_graph, embeddings, similarity, input_seq, 
                                 start_bucket=start_bucket,
                                 end_bucket=end_bucket,
                                 h_lap_pos_enc=lap_pos_enc,
                                 h_wl_pos_enc=wl_pos_enc)
        loss = model.loss(forecast, output_seq)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(forecast, output_seq)
        nb_data += output_seq.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    
    return epoch_loss, epoch_train_mae, optimizer

def evaluate_network(model, device, data_loader, event_graph, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0
    with torch.no_grad():
        event_graph = event_graph.to(device)
        for iter, (input_seq, output_seq, start_bucket) in enumerate(data_loader):
            input_seq_len = input_seq.size()[0]
            output_seq_len = output_seq.size()[0]
            end_bucket = start_bucket + (input_seq_len + output_seq_len) - 1
            input_seq = input_seq.to(device)
            output_seq = output_seq.to(device)
            embeddings = event_graph.ndata['feat'].to(device)
            similarity = event_graph.edata['feat'].to(device)
            try:
                lap_pos_enc = event_graph.ndata['lap_pos_enc'].to(device)
            except:
                lap_pos_enc = None
            
            try:
                wl_pos_enc = event_graph.ndata['wl_pos_enc'].to(device)
            except:
                wl_pos_enc = None
            forecast = model.forward(device, event_graph, embeddings, similarity,
                                     start_bucket=start_bucket,
                                     end_bucket=end_bucket,
                                     h_lap_pos_enc=lap_pos_enc, 
                                     h_wl_pos_enc=wl_pos_enc)
            loss = model.loss(forecast, output_seq)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(forecast, output_seq)
            nb_data += output_seq.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        
    return epoch_test_loss, epoch_test_mae
