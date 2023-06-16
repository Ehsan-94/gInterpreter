import argparse
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from math import sqrt
import math

from torch_geometric.datasets import TUDataset

import torch as th
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm
from sklearn import metrics
from scipy.spatial.distance import hamming
import statistics
import pandas
from time import perf_counter
from IPython.core.display import deepcopy
from torch_geometric.nn import MessagePassing
import copy
from torch.nn import ReLU, Sequential
from torch import sigmoid
from itertools import chain
from time import perf_counter
from torch_geometric.data import Data, Batch, Dataset
from functools import partial
from torch_geometric.utils import to_networkx
from torch_geometric.utils import remove_self_loops
from typing import Callable, Union, Optional
#from torch_geometric.utils.num_nodes import maybe_num_nodes
import networkx as nx
from typing import List, Tuple, Dict
from collections import Counter
import statistics
import tqdm
import csv


from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch_geometric.nn as gnn
import GCN_plus_GAP as Graph_Network
import CF2Explainer_Method




class CF2_Explaination(object):
    def __init__(self, Model_Name, your_dataset, explainer_epochs, fix_exp, classifier_load_index, input_dim, hid_dim,
                 output_dim, DataSet_name, importance_threshold):
        super(CF2_Explaination, self).__init__()

        self.Model_Name = Model_Name
        self.Task_name = 'Graph Classification'
        self.Explainability_name = "CF2"
        self.DataSet_name = DataSet_name
        self.GNN_Model = self.load_model(Task_name=self.Task_name, Explainability_name=self.Explainability_name,
                                         Model_Name=Model_Name,
                                         classifier_load_index=classifier_load_index, input_dim=input_dim,
                                         hid_dim=hid_dim, output_dim=output_dim)
        # self.GNN_Model = Graph_Network.GCN_plus_GAP(model_name='GCN_plus_GAP', model_level='graph', input_dim=7, hidden_dim=7, output_dim=2, num_hid_layers=2, Bias=True, act_fun='eLu', Weight_Initializer=1, dropout_rate=0.1)
        self.GNN_Model.eval()

        self.your_dataset = your_dataset
        self.explainer_epochs = explainer_epochs

        self.gamma_coeff = 0.5
        self.lambda_coeff = 10
        self.alpha_coeff = 0.6
        self.mask_threshold = 0.5
        self.explainer_lr = 0.002
        self.explainer_weight_decay = 0.005
        self.importance_threshold = importance_threshold

        if fix_exp:
            self.fix_exp = fix_exp * 2
        else:
            self.fix_exp = None
        t1 = perf_counter()
        correct_adj_mask = self.explain_nodes_gnn_stats(category='correct')[0]
        self.it_took = perf_counter()-t1
        #print("correct_adj_mask: ", correct_adj_mask)

        index_list = []
        adding = []
        banned = []
        for i, (start, target) in enumerate(zip(your_dataset[0].edge_index[0], your_dataset[0].edge_index[1])):
            if [start, target] not in banned:
                adding.append([start, target])
                index_list.append(i)
                banned.append([target, start])

        self.saliency_maps = {}
        self.importance_dict = {}
        edge_masks = correct_adj_mask.tolist()
        for i in range(len(index_list)):
            self.saliency_maps[i] = edge_masks[index_list[i]]

        for i in range(len(index_list)):
            self.importance_dict[i] = (correct_adj_mask>self.importance_threshold).tolist()[index_list[i]]
        #print(correct_adj_mask[0]>0.5)

        #print("your_dataset: ", len(your_dataset[0].edge_index[0]))
        #print("len(self.importance_dict): ", len(self.importance_dict))
        #print("self.impostance_dict: ", self.importance_dict)
        #print("self.saliency_maps: ", self.saliency_maps)


    def load_model(self, Task_name, Explainability_name, Model_Name, classifier_load_index, input_dim, hid_dim,
                   output_dim):

        if classifier_load_index != 0:
            GNN_Model, optimizer, classifier_load_index = self.loading_config(Task_name=Task_name,
                                                                              Explainability_name=Explainability_name,
                                                                              Model_Name=Model_Name,
                                                                              classifier_load_index=classifier_load_index,
                                                                              input_dim=input_dim, hid_dim=hid_dim,
                                                                              output_dim=output_dim)
            return GNN_Model
        else:
            GNN_Model = Graph_Network.GCN_plus_GAP(model_name=Model_Name, model_level='graph', input_dim=input_dim,
                                                   hidden_dim=hid_dim, output_dim=output_dim, num_hid_layers=2,
                                                   Bias=True, act_fun='eLu', Weight_Initializer=1, dropout_rate=0.1)
            return GNN_Model

    def loading_config(self, Task_name, Explainability_name, Model_Name, classifier_load_index, input_dim, hid_dim,
                       output_dim):
        GNN_Model = Graph_Network.GCN_plus_GAP(model_name=Model_Name, model_level='graph', input_dim=input_dim,
                                               hidden_dim=hid_dim, output_dim=output_dim,
                                               num_hid_layers=2, Bias=True, act_fun='eLu', Weight_Initializer=1,
                                               dropout_rate=0.1)
        optimizer = torch.optim.Adam(params=GNN_Model.parameters(), lr=0.001, weight_decay=1e-6)
        checkpoint = torch.load(Model_Name + " " + Explainability_name + " " + Task_name + " " + self.DataSet_name + " " + str(classifier_load_index) + ".pt")
        GNN_Model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return GNN_Model, optimizer, epoch

    def clear_masks(self, model):
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.explain = False
                module._edge_mask = None
                module._loop_mask = True
                module._apply_sigmoid = False
        return module


    def explain(self, graph, original_pred, category):

        cf2_explainer = CF2Explainer_Method.CF2_Explainer(graph=graph, GNN_Model=self.GNN_Model, reset_mask_to_None=True)
        cf2_explainer_optimizer = torch.optim.Adam(cf2_explainer.parameters(), lr=self.explainer_lr,
                                                   weight_decay=self.explainer_weight_decay)

        cf2_explainer.train()
        for epoch in range(self.explainer_epochs):
            cf2_explainer.zero_grad()
            pred_factual, pred_c_factual = cf2_explainer.forward()

            if category == 'correct':
                pred_factual = pred_factual.argmax(dim=1)
                pred_c_factual = pred_c_factual.argmax(dim=1)
            else:
                pred_factual = pred_factual.argmin(dim=1)
                pred_c_factual = pred_c_factual.argmin(dim=1)
            bpr1, bpr2, l1, loss = cf2_explainer.loss(pred_factual, pred_c_factual, original_pred, self.gamma_coeff,
                                                      self.lambda_coeff, self.alpha_coeff)

            loss.backward()
            cf2_explainer_optimizer.step()

        masked_adj = cf2_explainer.get_masked_adj()
        masked_adj = cf2_explainer.get_masked_adj()

        complexity_cost = len(masked_adj[masked_adj > self.mask_threshold])
        # filtered_masked_adj = (masked_adj > self.mask_threshold) * 1

        # exp_num = new_edge_num #new_edge_num / 2
        return masked_adj, complexity_cost

    def explain_nodes_gnn_stats(self, category):
        exp_dict = {}  # {'gid': masked_adj, 'gid': mask_adj}
        num_dict = {}  # {'gid': complexity_cost, 'gid': complexity_cost}
        for i, graph in enumerate(self.your_dataset):
            self.clear_masks(self.GNN_Model)
            Output_of_Hidden_Layers, pooling_layer_output, ffn_output, original_pred = self.GNN_Model(graph)
            original_pred = original_pred.argmax(dim=1)
            original_label = graph.y

            masked_adj, complexity_cost = self.explain(graph, original_pred, category)
            print("graph has: ", len(graph.edge_index[0]), "edge complexity_cost: ", complexity_cost)
            exp_dict[i] = masked_adj
            num_dict[i] = complexity_cost
        print('average number of exps:', sum(num_dict.values()) / len(num_dict.keys()))
        return exp_dict


dataset = TUDataset(root='data/TUDataset', name='MUTAG')

cf2_explanation = CF2_Explaination(Model_Name="GCN_plus_GAP", your_dataset=[dataset[0]], explainer_epochs=1000,
                                   fix_exp=None, classifier_load_index=200, DataSet_name="MUTAG",
                                   input_dim=7, hid_dim=7, output_dim=2, importance_threshold=0.5)
#t1 = perf_counter()
#correct_adj_mask = cf2_explanation.explain_nodes_gnn_stats(category='correct')
#t2 = perf_counter()
#print("adj_mask for correct[0]: ", correct_adj_mask[0])


#attribution_time = (t2 - t1) / len(test_dataset)
#print("AVG Time: ", (t2 - t1) / len(test_dataset))






