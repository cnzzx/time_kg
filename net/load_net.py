"""
    Utility file to select GraphNN model as
    selected by the user
"""
from net.gt_mlp import GraphTransformerNet

def GraphTransformer(net_params):
    return GraphTransformerNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GraphTransformer': GraphTransformer
    }
        
    return models[MODEL_NAME](net_params)