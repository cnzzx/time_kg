import torch
import pickle
import torch.utils.data
import time
import os
import sys
sys.path.append(os.path.abspath('.'))
import data.load as load
import data.event.process as eprocess
import numpy as np

import csv

import dgl

from scipy import sparse as sp
import numpy as np
import networkx as nx
import hashlib


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    
    return g

def wl_positional_encoding(g):
    """
        WL-based absolute positional embedding 
        adapted from 
        
        "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
        Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
        https://github.com/jwzhanggy/Graph-Bert
    """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1


    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1
        
    g.ndata['wl_pos_enc'] = torch.LongTensor(list(node_color_dict.values()))
    return g


class FinancialDataset(torch.utils.data.Dataset):
    """
        Time series dataset including multiple series slices.
    """
    def __init__(self, input_seqs, output_seqs):
        self.input_seqs = input_seqs
        self.output_seqs = output_seqs
        self.n_samples = input_seqs.size()[0]

    def __getitem__(self, index):
        return self.input_seqs[index], self.output_seqs[index]
    
    def __len__(self):
        return self.n_samples


class FinancialDatapack():
    """
        Include trainning, validation and testing series slices
        and event graph.
    """
    def __init__(self, name, input_step_len=5, output_step_len=10, test_ratio=0.2, val_ratio=0.2):
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_path = 'data/datasets/' + name + '.csv'
        input_seqs, output_seqs = load.make_dataset(load.load_data(data_path, name), input_step_len, output_step_len, k=0.25)

        # shuffling
        # avoid data distribution offset
        n_samples = len(input_seqs)
        for shuffle_idx in range(1, n_samples):
            another_idx = np.random.randint(shuffle_idx)
            temp = input_seqs[another_idx].copy()
            input_seqs[another_idx] = input_seqs[shuffle_idx].copy()
            input_seqs[shuffle_idx] = temp.copy()
            temp = output_seqs[another_idx].copy()
            output_seqs[another_idx] = output_seqs[shuffle_idx].copy()
            output_seqs[shuffle_idx] = temp.copy()

        self.num_vertex_type = 1
        self.num_edge_type = 1

        # divide the dataset into training, validation and testing sets
        train_input = torch.tensor(input_seqs[:int(n_samples*(1-test_ratio)*(1-val_ratio))])
        val_input = torch.tensor(input_seqs[int(n_samples*(1-test_ratio)*(1-val_ratio)):int(n_samples*(1-test_ratio))])
        test_input = torch.tensor(input_seqs[int(n_samples*(1-test_ratio)):])

        train_output = torch.tensor(output_seqs[:int(n_samples*(1-test_ratio)*(1-val_ratio))])
        val_output = torch.tensor(output_seqs[int(n_samples*(1-test_ratio)*(1-val_ratio)):int(n_samples*(1-test_ratio))])
        test_output = torch.tensor(output_seqs[int(n_samples*(1-test_ratio)):])

        self.train = FinancialDataset(train_input, train_output)
        self.val = FinancialDataset(val_input, val_output)
        self.test = FinancialDataset(test_input, test_output)

        # load graph data
        dates, embeddings, graph_adj, graph_sim = eprocess.get_event_info()

        # Build DGL graph based on those infos.
        # Create the DGL graph.
        n_nodes = graph_adj.size()[0]  # 930 for the financial events collected
        self.event_dgl = dgl.DGLGraph()
        self.event_dgl.add_nodes(n_nodes)

        # Build the graph based on graph_adj.
        for u in range(n_nodes):
            for v in range(n_nodes):
                if graph_adj[u, v] > 0:
                    self.event_dgl.add_edge(u, v)
        
        # Set vertex and edge features.
        self.event_dgl.ndata['feat'] = embeddings
        self.event_dgl.ndata['date'] = dates
        self.event_dgl.edata['feat'] = graph_sim

        print('train, test, val sizes :',self.train_input.size()[0],self.test_input.size()[0],self.val_input.size()[0])
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        # labels = torch.tensor(np.array(labels)).unsqueeze(1)
        labels = torch.tensor(labels).unsqueeze(1)
        batched_graph = dgl.batch(graphs)
        
        return batched_graph, labels
    
    def _add_laplacian_positional_encodings(self, pos_enc_dim):
        # Graph positional encoding v/ Laplacian eigenvectors
        self.event_dgl = laplacian_positional_encoding(self.event_dgl, pos_enc_dim)

    def _add_wl_positional_encodings(self):
        # WL positional encoding from Graph-Bert, Zhang et al 2020.
        self.event_dgl = wl_positional_encoding(self.event_dgl)
