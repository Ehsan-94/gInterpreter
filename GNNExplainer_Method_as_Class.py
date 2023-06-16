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
import GNNExplainer_Method as gnnexplainer_method



class GNNExplainer_GC(object):
    def __init__(self, task, method, model_name, graph, importance_threshold, load_index, input_dim, hid_dim, output_dim,
                 normalize_coeff, DataSet_name):
        self.normalize_coeff = normalize_coeff
        self.importance_threshold = importance_threshold
        self.GNN_model = self.load_model(task, method, model_name, load_index=load_index, input_dim=input_dim, hid_dim=hid_dim,
                                         output_dim=output_dim, DataSet_name=DataSet_name)
        self.gnnexplainer = gnnexplainer_method.GNNExplainer(self.GNN_model, 200, 0.001)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.importance_dict = {}
        start_time = perf_counter()
        self.new_graph, self.saliency_maps = self.drop_important_nodes(graph, self.importance_threshold)
        self.it_took = perf_counter()-start_time
    def load_model(self, task, method, model_name, load_index, input_dim, hid_dim, output_dim, DataSet_name):

        if load_index != 0:
            GNN_model, optimizer, load_index = self.loading_config(task, method, model_name, load_index, input_dim,
                                                                   hid_dim, output_dim, DataSet_name)
            return GNN_model
        else:
            GNN_model = Graph_Network.GCN_plus_GAP(model_name=model_name, model_level='graph', input_dim=input_dim,
                                                   hidden_dim=hid_dim, output_dim=output_dim, num_hid_layers=2, Bias=True,
                                                   act_fun='eLu', Weight_Initializer=1, dropout_rate=0.1)
            return GNN_model

    def loading_config(self, task, method, model_name, load_index, input_dim, hid_dim, output_dim, DataSet_name):
        GNN_model = Graph_Network.GCN_plus_GAP(model_name='GCN_plus_GAP', model_level='graph', input_dim=input_dim, hidden_dim=hid_dim, output_dim=output_dim, num_hid_layers=2, Bias=True, act_fun='eLu', Weight_Initializer=1, dropout_rate=0.1)
        optimizer = torch.optim.Adam(GNN_model.parameters(), lr=0.001, weight_decay=1e-6)
        checkpoint = torch.load(str(model_name) + " " + str(method) + " " + str(task) + " " + str(DataSet_name) + " " + str(load_index)+".pt")
        GNN_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return GNN_model, optimizer, epoch

    def loss_calculations(self, preds, gtruth):
        loss_per_epoch = self.criterion(preds, gtruth)
        return loss_per_epoch


    def is_salient(self, score, importance_threshold):
        if importance_threshold == score == 0:
            return True
        if importance_threshold == score == 1:
            return False
        if importance_threshold < score:
            return True
        else:
            return False
    def get_attribution_scores(self, node_mask, g):
        attributed_graph = []
        attributed_graph_normalized = []
        attributed_graph_standardized = []
        for node in g.x:
            #print(sum(np.multiply(node, node_mask)))
            attributed_graph.append(sum(np.multiply(node, node_mask)))
        for score in attributed_graph:
            attributed_graph_normalized.append(abs(((score-min(attributed_graph))*self.normalize_coeff/(max(attributed_graph)-min(attributed_graph)))).tolist())
        for score in attributed_graph_normalized:
            attributed_graph_standardized.append(abs(score / max(attributed_graph_normalized)))
        return attributed_graph_standardized

    def is_salient(self, score, importance_threshold):
        if importance_threshold == score == 0:
            return True
        if importance_threshold == score == 1:
            return False
        if importance_threshold < score:
            return True
        else:
            return False
    def edges_attribution_scores(self, edge_mask, graph):
        edges_saliency = []

        index_list = []
        adding = []
        banned = []
        for i, (start, target) in enumerate(zip(graph.edge_index[0], graph.edge_index[1])):
            if [start, target] not in banned:
                adding.append([start, target])
                index_list.append(i)
                banned.append([target, start])
                edges_saliency.append(edge_mask[i].tolist())

        norm_edges = []
        standard_edges = []
        for score in edges_saliency:
            norm_edges.append(abs(((score-min(edges_saliency))*self.normalize_coeff/(max(edges_saliency)-min(edges_saliency)))))
        for score in norm_edges:
            standard_edges.append(abs(score / max(norm_edges)))
        #print("standard_edges: ", standard_edges)
        edges_saliency_maps = {}
        edges_importance_dict = {}
        for i, edge in enumerate(standard_edges):
            edges_importance_dict[i] = self.is_salient(edge, self.importance_threshold)
            edges_saliency_maps[i] = edge
        return edges_importance_dict, edges_saliency_maps
    def drop_important_nodes(self, graphs, importance_threshold):


        #print(len(correct_node_mask), correct_node_mask)

        occluded_GNNgraph_list = []
        Standard_GNNExplainer_attribution_scores = []
        #print("graphs: ",graphs)

        for i, g in enumerate(graphs):
            correct_node_mask, correct_edge_mask = self.gnnexplainer(g, "correct")
            #print("correct_edge_mask: ", correct_edge_mask)
            self.edges_importance_dict, self.edges_saliency_maps = self.edges_attribution_scores(correct_edge_mask, g)
            incorrect_node_mask, incorrect_edge_mask = self.gnnexplainer(g, "incorrect")
            attributed_graph_normalized = self.get_attribution_scores(correct_node_mask, g)
            Standard_GNNExplainer_attribution_scores.append(attributed_graph_normalized)
            sample_graph = deepcopy(g)
            graph_dict = {}
            for j in range(len(sample_graph.x)):
                if self.is_salient(attributed_graph_normalized[j], importance_threshold):
                    # print("before: ", sample_graph.x[j])
                    sample_graph.x[j][:] = 0
                    # print(torch.zeros_like(sample_graph.x[j]))
                    # print("manipulated: ",sample_graph.x[j])
                    graph_dict[j] = True
                else:
                    graph_dict[j] = False
            self.importance_dict[i] = graph_dict
            occluded_GNNgraph_list.append(sample_graph)
        #print(sample_graph.x)

        #print("Standard_GNNExplainer_attribution_scores: ", Standard_GNNExplainer_attribution_scores)
        return occluded_GNNgraph_list, Standard_GNNExplainer_attribution_scores


#dataset = TUDataset(root='data/TUDataset', name='MUTAG')

#new_output = GNNExplainer_GC(model_name="GCN_plus_GAP", task="Graph Classification", method="GNNExplainer", graph=[dataset[0]],
#                             importance_threshold=0.5, load_index=200, input_dim=len(dataset[0].x[0]), hid_dim=7,
#                             output_dim=2, normalize_coeff=100, DataSet_name="MUTAG")
#print(new_output.new_graph[-1].x, dataset[-1].x)
#print(new_output.it_took)
#print("importance_dict: ", new_output.importance_dict)
#print("saliency_maps: ", new_output.saliency_maps)
#print("edges_importance_dict: ", new_output.edges_importance_dict)
#print("edges_saliency_maps: ", new_output.edges_saliency_maps)
