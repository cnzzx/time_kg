import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
Graph Transformer with edge features
"""
from layers.graph_transformer_edge_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout


def get_time_bucket(date, time_gap):
    return (date-1) // time_gap + 1


class GT_MLPNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        max_wl_role_index = 37 # this is maximum graph size in the dataset
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        
        self.fn_embedding_h = nn.Linear(out_dim, 1)  # embedding the final event vector to scalar
        self.fn_embedding_xh = nn.Linear(2, 1)

        self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem        
        
    def forward(self, g, h, e, x, start_bucket, end_bucket, k=0.25, time_gap=7, h_lap_pos_enc=None, h_wl_pos_enc=None):
        """
        g: the graph structure
        g.ndata: the vertex feature dict
                 There should be the:
                 - start date of one event (g.ndata['sd'])
        g.edata: the edge feature dict
        h: vertex features
        e: edge features. For financial events, these represents the semantic similarity.
        x: the time series
        start_bucket: the start time bucket of input time series
        end_bucket: the end time bucket of output time series
                    the length of x is less than (end_bucket-start_bucket+1)
        """
        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)
        e = self.embedding_e(e)   
        
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h

        # do the influence computing
        # this module makes SGDs almost the only proper optimization method
        
        input_seq_len = x.size()[0]
        output_seq_len = (end_bucket-start_bucket+1) - input_seq_len
        # Only consider events whose influence can cover the time slice.
        event_influence = torch.zeros((end_bucket - start_bucket + 1), requires_grad=True)
        event_dates = g.ndata['sd']
        n_events = event_dates.size()[0]
        influence_span = int(1 / k)
        for event_idx in range(n_events):
            event_bucket = get_time_bucket(event_dates[event_idx], time_gap)
            if event_bucket + influence_span - 1 < start_bucket:
                continue
            if event_bucket + influence_span - 1 > end_bucket:
                break
            # Binary search might be faster but I guess it's not so important
            # even the events considered are very sparse for one sample.
            for inf_dis in range(influence_span):
                event_influence[event_bucket + inf_dis - start_bucket] += (1 - k * inf_dis) * h[event_idx]
                # linear degradation
        enhanced_series = torch.cat((x, event_influence[:input_seq_len]), axis=0)
        enhanced_series = self.fn_embedding_xh(enhanced_series)
        final_feat = torch.cat((enhanced_series, event_influence[input_seq_len:]), axis=1)
        return self.MLP_layer(final_feat)
        
    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss
