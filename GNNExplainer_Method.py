import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas
from torch_geometric.datasets import TUDataset
import torch
from torch_geometric.loader import DataLoader
import GCN_plus_GAP as Graph_Network
from copy import deepcopy
import numpy as np
from time import perf_counter
from scipy.special import softmax
from statistics import mean
import torch_geometric
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from IPython.core.display import deepcopy
from torch_geometric.nn import MessagePassing
import copy
from math import sqrt

loss_fn = F.binary_cross_entropy_with_logits


class GNNExplainer:
    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(self, GNN_Model, Exp_Epoch, Exp_lr):

        self.GNN_Model = GNN_Model
        self.loss_fn = F.binary_cross_entropy_with_logits

        self.explainer_epochs = Exp_Epoch
        self.explainer_lr = Exp_lr
        self.node_mask = self.edge_mask = None
        self.softmax = nn.Softmax(dim=-1)

    def initialize_masks(self, graph_to_explain):
        # (N, F), E = (len(graph_to_explain.x), len(graph_to_explain.x[0])), len(graph_to_explain.edge_index[1])
        (N, F), E = (1, len(graph_to_explain.x[0])), len(graph_to_explain.edge_index[1])
        std = 0.1
        self.node_mask = Parameter(torch.randn(N, F) * std, requires_grad=True)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = Parameter(torch.randn(E) * std, requires_grad=True)

        for module in self.GNN_Model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask
                module._apply_sigmoid = True

    def explainer_loss(self, By_Perturbation_predicted_label, predicted_label):
        By_Perturbation_predicted_label = By_Perturbation_predicted_label.to(torch.float32)
        predicted_label = predicted_label.to(torch.float32)
        # print('By_Perturbation_predicted_label', By_Perturbation_predicted_label)
        # print('predicted_label', predicted_label)
        loss_in_explainer_stage = self.loss_fn(By_Perturbation_predicted_label, predicted_label)

        m = self.edge_mask.sigmoid()
        edge_reduce = getattr(torch,
                              self.coeffs['edge_reduction'])  ######.         MARGINALIZE Over All Feature Subsets
        loss_in_explainer_stage = loss_in_explainer_stage + self.coeffs['edge_size'] * edge_reduce(m)
        ent = -m * torch.log(m + self.coeffs['EPS']) - (1 - m) * torch.log(1 - m + self.coeffs['EPS'])
        loss_in_explainer_stage = loss_in_explainer_stage + self.coeffs['edge_ent'] * ent.mean()

        m = self.node_mask.sigmoid()  ######.         Element-wise Entropy for structural and node feature masks to be discrete
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss_in_explainer_stage = loss_in_explainer_stage + self.coeffs['node_feat_size'] * node_feat_reduce(m)
        ent = -m * torch.log(m + self.coeffs['EPS']) - (1 - m) * torch.log(1 - m + self.coeffs['EPS'])
        loss_in_explainer_stage = loss_in_explainer_stage + self.coeffs['node_feat_ent'] * ent.mean()

        return loss_in_explainer_stage

    def explainer_train_step(self, graph_to_explain, predicted_label):
        new_graph_by_masks = graph_to_explain

        parameters = [self.node_mask]
        parameters.append(self.edge_mask)

        explainer_optimizer = torch.optim.Adam(parameters, lr=self.explainer_lr)

        for i in range(self.explainer_epochs):
            explainer_optimizer.zero_grad()

            h_node = graph_to_explain.x * self.node_mask.sigmoid()
            new_graph_by_masks.x = h_node

            By_Perturbation_Output_of_Hidden_Layers, By_Perturbation_pooling_layer_output, By_Perturbation_gcn_model_output, By_Perturbation_soft_y_hat = self.GNN_Model(new_graph_by_masks)
            By_Perturbation_predicted_label = By_Perturbation_soft_y_hat.argmax(dim=-1)
            #final_GNN_layer_output, sortpooled_embedings, output_conv1d_1, maxpooled_output_conv1d_1, output_conv1d_2, to_dense, output_h1, dropout_output_h1, output_h2, softmaxed_h2 = self.GNN_Model(
            #    graph_to_explain)
            #By_Perturbation_predicted_label = softmaxed_h2.argmax(dim=-1)

            loss = self.explainer_loss(By_Perturbation_predicted_label, predicted_label)

            # print(loss)

            loss.backward()
            explainer_optimizer.step()
            # print(self.edge_mask)
            # print(self.node_mask)

    def post_process_mask(self, mask, apply_soft):
        # print(apply_sig)
        if apply_soft:
            mask = mask.detach()
            mask = self.softmax(mask)
            return mask
        else:
            return mask

    def train_explainer(self, graph_to_explain, class_type):
        # new_graph_by_masks = graph_to_explain
        with torch.no_grad():
            Output_of_Hidden_Layers, pooling_layer_output, ffn_output, soft = self.GNN_Model(graph_to_explain)
            #final_GNN_layer_output, sortpooled_embedings, output_conv1d_1, maxpooled_output_conv1d_1, output_conv1d_2, to_dense, output_h1, dropout_output_h1, output_h2, softmaxed_h2 = self.GNN_Model(
            #    graph_to_explain)
        # predicted_label = softmaxed_h2.argmax(dim=-1)
        # print("before softmax: ", soft)
        if class_type == "correct":
            predicted_label = soft.argmax(dim=-1)
        else:
            predicted_label = soft.argmin(dim=-1)
            # print("after softmax: ", predicted_label)

        self.initialize_masks(graph_to_explain)
        self.explainer_train_step(graph_to_explain, predicted_label)

        node_mask = self.post_process_mask(self.node_mask, True)
        edge_mask = self.post_process_mask(self.edge_mask, True)
        return node_mask, edge_mask

    def __call__(self, graph_to_explain, class_type):
        new_graph_by_masks = deepcopy(graph_to_explain.detach())
        node_mask, edge_mask = self.train_explainer(new_graph_by_masks, class_type)
        node_mask = torch.squeeze(node_mask)
        # print("Node Mask: ", node_mask)
        # print("Edge Mask: ", edge_mask)
        return node_mask, edge_mask