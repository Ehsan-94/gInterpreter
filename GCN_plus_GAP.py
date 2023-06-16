from ast import mod
#from torch_geometric.sampler.neighbor_sampler import torch_geometric


import argparse
import os
import torch as th
import torch
import torch.nn as nn
from torch_sparse import SparseTensor
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.nn import Linear
#from sklearn.model_selection import train_test_split
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer
import torch_geometric.nn as gnn


class GlobalMeanPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return gnn.global_mean_pool(x, batch)
################################################################################
class IdenticalPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return x

################################################################################
class GCN_plus_GAP(torch.nn.Module):
    def __init__(self, model_name, model_level, input_dim, hidden_dim, output_dim, num_hid_layers, Bias, act_fun, Weight_Initializer, dropout_rate,
                 pred_hidden_dims=[], concat=True, bn=True,
                 add_self=False, args=None):
        if model_name == 'GCN_plus_GAP':
            super(GCN_plus_GAP, self).__init__()
            self.input_dim = input_dim
            print ('GCN_plus_GAP Input_Dimension:', self.input_dim)

            self.hidden_dim = hidden_dim
            print ('GCN_plus_GAP Hidden_Dimension:', self.hidden_dim)

            self.output_dim = output_dim
            print ('GCN_plus_GAP Output_Dimension:', self.output_dim)

            self.num_hid_layers = num_hid_layers
            print ('GCN_plus_GAP Number_of_Hidden_Layers:', self.num_hid_layers)


            self.args = args
            if act_fun == 'ReLu':
                self.act_fun = F.relu
                print('ReLu is Selected.')
            elif act_fun == 'eLu':
                self.act_fun = nn.functional.elu
                print('eLu is Selected.')

            self.GConvs = torch.nn.ModuleList()

            for layer in range(self.num_hid_layers):
                self.GConvs.append(GCNConv(self.hidden_dim, self.hidden_dim, bias=Bias))

            #print('len(self.GConvs):', len(self.GConvs))
            self.dropout = nn.Dropout(p=dropout_rate)

            if model_level == 'node':
                self.readout = IdenticalPool()
            else:
                self.readout = GlobalMeanPool()

            self.ffn = nn.Linear(self.hidden_dim, self.output_dim, bias=Bias)

            mean = 0
            std = 0.1
            self.initialize_weights(Weight_Initializer, Bias, mean, std)



        else:
            print('This is 2GCN_plus_GAP Model, please type its name well...')

    def initialize_weights(model, Weight_Initializer, Bias, mean, std):
        # 1. Xavier Normal_.  2. Kaiming Normal_.  3. Uniform (0,0.1std)
        if Weight_Initializer == 1:                                             #.      1. Xavier Normal_.
            for i,layers in enumerate(model.children()):
                if isinstance(layers, torch.nn.ModuleList):
                    for j, layer in enumerate(layers.modules()):
                        if isinstance(layer, GCNConv):
                            torch.nn.init.xavier_normal_(layer.lin.weight)
                            if Bias:
                                layer.bias.data.zero_()
                        else: 
                            pass
                if isinstance(layers, torch.nn.Linear):
                    torch.nn.init.xavier_normal_(layers.weight)
                    if Bias:
                        layers.bias.data.zero_()

                elif isinstance(layers, (GlobalMeanPool)):
                    pass
                elif isinstance(layers, (IdenticalPool)):
                    pass

        if Weight_Initializer == 2:                                             #.      2. Kaiming Normal_.
            for i,layers in enumerate(model.children()):
                if isinstance(layers, torch.nn.ModuleList):
                    for j, layer in enumerate(layers.modules()):
                        if isinstance(layer, GCNConv):
                            torch.nn.init.kaiming_normal_(layer.lin.weight)
                            if Bias:
                                layer.bias.data.zero_()
                        else: 
                            pass
                if isinstance(layers, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(layers.weight)
                    if Bias:
                        layers.bias.data.zero_()
                elif isinstance(layers, (GlobalMeanPool)):
                    pass
                elif isinstance(layers, (IdenticalPool)):
                    pass
                            
        if Weight_Initializer == 3:                                             #.      3. Uniform (0,0.1std)
            for i,layers in enumerate(model.children()):
                if isinstance(layers, torch.nn.ModuleList):
                    for j, layer in enumerate(layers.modules()):
                        if isinstance(layer, GCNConv):
                            torch.nn.init.normal_(layer.lin.weight.data, mean, std)
                            if Bias:
                                layer.bias.data.zero_()
                        else: 
                            pass
                if isinstance(layers, torch.nn.Linear):
                    torch.nn.init.normal_(layers.weight, mean, std)
                    if Bias:
                        layers.bias.data.zero_()
                elif isinstance(layers, (GlobalMeanPool)):
                    pass
                elif isinstance(layers, (IdenticalPool)):
                    pass



    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        Output_of_Hidden_Layers = []
        for i in range(self.num_hid_layers):
            x = self.GConvs[i](x, edge_index)
            x = self.act_fun(x)
            x = self.dropout(x)
            Output_of_Hidden_Layers.append(x)

        pooling_layer_output = self.readout(x, batch)
        ffn_output = self.ffn(pooling_layer_output)
        ffn_output = self.act_fun(ffn_output)

        #log_soft = F.log_softmax(ffn_output, dim=1)
        soft = F.softmax(ffn_output, dim=1)

        return Output_of_Hidden_Layers, pooling_layer_output, ffn_output, soft
        #return Output_of_Hidden_Layers, pooling_layer_output, ffn_output, log_soft, soft

GNN_Model = GCN_plus_GAP(model_name='GCN_plus_GAP', model_level='graph', input_dim=7,
                         hidden_dim=7, output_dim=2, num_hid_layers=2, Bias=True, 
                         act_fun='eLu', Weight_Initializer=1, dropout_rate=0.1)
print('===================================================================================')
print(GNN_Model)
print('===================================================================================')






#dataset = TUDataset(root='data/TUDataset', name='MUTAG')

#from torch_geometric.loader import DataLoader
#batched_dataset = DataLoader(dataset, batch_size=23, shuffle=False)

#for batch in batched_dataset:
#    Output_of_Hidden_Layers, pooling_layer_output, ffn_output, soft = GNN_Model(batch)
#    print("ffn_output: ", ffn_output)
#    #print("log_soft: ", log_soft)
#    print("soft: ", soft)
#    break