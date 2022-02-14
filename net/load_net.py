"""
    Utility file to select GraphNN model as
    selected by the user
"""
from net.graph_transformer_net import GraphTransformerNet
from net.gt_mlp import GT_MLPNet

def GraphTransformer(net_params):
    return GraphTransformerNet(net_params)

def GT_MLP(net_params):
    return GT_MLPNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GraphTransformer': GraphTransformer,
        'GT_MLP': GT_MLP
    }
    return models[MODEL_NAME](net_params)