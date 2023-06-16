import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas
from torch_geometric.datasets import TUDataset
import torch
from torch_geometric.loader import DataLoader
#import Model_Loading as gcn_2l_model
import GCN_plus_GAP as Graph_Network
from copy import deepcopy
import numpy as np
from time import perf_counter
from scipy.special import softmax
from statistics import mean
from numpy import exp


class GuidedBP_GC(object):
    def __init__(self, task, method, model_name, graph, importance_threshold, load_index, input_dim, hid_dim, output_dim,
                 normalize_coeff):
        self.normalize_coeff = normalize_coeff
        self.GCN_model = self.load_model(task, method, model_name, load_index=load_index, input_dim=input_dim,
                                         hid_dim=hid_dim, output_dim=output_dim)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.importance_dict = {}
        start_time = perf_counter()
        self.new_graph, self.saliency_maps = self.drop_important_nodes(graph, importance_threshold)
        self.it_took = perf_counter()-start_time
    def load_model(self, task, method, model_name, load_index, input_dim, hid_dim, output_dim):

        if load_index != 0:
            GCN_model, optimizer, load_index = self.loading_config(task, method, model_name, load_index, input_dim, hid_dim, output_dim)
            return GCN_model
        else:
            GCN_model = Graph_Network.GCN_plus_GAP(model_name=model_name, model_level='graph', input_dim=input_dim,
                                                   hidden_dim=hid_dim, output_dim=output_dim, num_hid_layers=2, Bias=True,
                                                   act_fun='eLu', Weight_Initializer=1, dropout_rate=0.1)
            return GCN_model





    def loading_config(self, task, method, model_name, load_index, input_dim, hid_dim, output_dim):
        GCN_model = Graph_Network.GCN_plus_GAP(model_name=model_name, model_level='graph', input_dim=input_dim,
                                               hidden_dim=hid_dim, output_dim=output_dim, num_hid_layers=2, Bias=True,
                                               act_fun='eLu', Weight_Initializer=1, dropout_rate=0.1)
        DataSet_name = "MUTAG"
        optimizer = torch.optim.Adam(GCN_model.parameters(), lr=0.001, weight_decay = 1e-6)
        checkpoint = torch.load(str(model_name) + " " + str(method) + " " + str(task) + " " + str(DataSet_name) + " " + str(load_index)+".pt")
        GCN_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return GCN_model, optimizer, epoch




    def loss_calculations(self, preds, gtruth):
        loss_per_epoch = self.criterion(preds, gtruth)
        return loss_per_epoch




    def remove_nones(self, sample_grads):
        sample_grads2 = []
        for item in sample_grads:
          Each_Graph = []
          for item2 in item:
            if item2 != None:
              Each_Graph.append(torch.tensor(item2.clone().detach().requires_grad_(True), requires_grad=True))
            else:
              Each_Graph.append(torch.tensor(0))
          sample_grads2.append(Each_Graph)

        return sample_grads2




    def compute_grad(self, model, graph, with_respect):
        post_conv1, post_conv2, out_readout, prediction = model(graph)
        #print(prediction)
        if with_respect == 1:
            loss = self.loss_calculations(prediction, graph.y)
            # print(loss)
        elif with_respect == 2:
            loss = self.loss_calculations(prediction, torch.tensor([0]))
            # print(loss)
        elif with_respect == 3:
            loss = self.loss_calculations(prediction, torch.tensor([1]))
            # print(loss)
        return torch.autograd.grad(loss, list(self.GCN_model.parameters()), allow_unused=True)




    def compute_sample_grads(self, model, test_dataset, with_respect):

        sample_grads = [self.compute_grad(model, graph, with_respect) for graph in test_dataset]
        sample_grads = self.remove_nones(sample_grads)
        sample_grads = zip(*sample_grads)
        sample_grads = [torch.stack(shards) for shards in sample_grads]
        return sample_grads

    def column_wise_addups(self, gradients):
        return np.sum(gradients, axis=1).tolist()

    def compute_guided_gradients(self, your_model, dataset):
        per_sample_grads_wrt_graph_label = self.compute_sample_grads(your_model, dataset, 1)
        per_sample_grads_wrt_class_zero = self.compute_sample_grads(your_model, dataset, 2)
        per_sample_grads_wrt_class_one = self.compute_sample_grads(your_model, dataset, 3)

        grads_wrt_graph_label = per_sample_grads_wrt_graph_label[1]
        guided_grads_wrt_graph_label = torch.maximum(torch.zeros_like(grads_wrt_graph_label), grads_wrt_graph_label)
        guided_grads_wrt_graph_label = guided_grads_wrt_graph_label.detach().tolist()

        grads_wrt_class_zero = per_sample_grads_wrt_class_zero[1]
        guided_grads_wrt_class_zero = torch.maximum(torch.zeros_like(grads_wrt_class_zero), grads_wrt_class_zero)
        guided_grads_wrt_class_zero = guided_grads_wrt_class_zero.detach().tolist()

        grads_wrt_class_one = per_sample_grads_wrt_class_one[1]
        guided_grads_wrt_class_one = torch.maximum(torch.zeros_like(grads_wrt_class_one), grads_wrt_class_one)
        guided_grads_wrt_class_one = guided_grads_wrt_class_one.detach().tolist()

        guided_grads_wrt_graph_label = self.column_wise_addups(guided_grads_wrt_graph_label)
        guided_grads_wrt_class_zero = self.column_wise_addups(guided_grads_wrt_class_zero)
        guided_grads_wrt_class_one = self.column_wise_addups(guided_grads_wrt_class_one)

        return guided_grads_wrt_graph_label, guided_grads_wrt_class_zero, guided_grads_wrt_class_one




    def saliency(self, input_graphs, graphs_gradients):
        #gradients = self.softmax(gradients)
        Graphs_new_gradients = []
        for graph_grads in graphs_gradients:
            new_gradients = []
            for dim in graph_grads:
                new_gradients.append(((dim-min(graph_grads))*self.normalize_coeff)/(max(graph_grads)-min(graph_grads)))
            Graphs_new_gradients.append(new_gradients)

        graphs_attributed_node_feat = []
        for input_graph, g_grads in zip(input_graphs, Graphs_new_gradients):
            attributed_node_feat = []
            for node_feat in input_graph.x:
                attributed_node_feat.append(np.multiply(node_feat, g_grads).tolist())
            graphs_attributed_node_feat.append(attributed_node_feat)

        graphs_attributed_nodes = []
        for graph_atts in graphs_attributed_node_feat:
            attributed_nodes = []
            for node_feat in graph_atts:
                attributed_nodes.append(sum(node_feat))
            graphs_attributed_nodes.append(attributed_nodes)
        #print("graphs_attributed_nodes: ", graphs_attributed_nodes)

        graph_nodes_importance_level = []
        for graph_nodes in graphs_attributed_nodes:
            nodes_importance_level = []
            for node_importance in graph_nodes:
                nodes_importance_level.append(node_importance/max(graph_nodes))
            graph_nodes_importance_level.append(nodes_importance_level)
        #print("graph_nodes_importance_level: ", graph_nodes_importance_level)

        return graph_nodes_importance_level



    def is_salient(self, index, score, importance_threshold):
        if importance_threshold == score == 0:
            return True
        if importance_threshold == score == 1:
            return False
        if importance_threshold < score:
            return True
        else:
            return False

    def standardize_by_softmax(self, saliencies):
        softmaxed_attributions = []
        for graph_saliency in saliencies:
            #print(graph_saliency)
            #print("this is the softmax: ",softmax(graph_saliency))
            softmaxed_attributions.append(softmax(graph_saliency).tolist())
        standard_attributions = []
        for soft_atts in softmaxed_attributions:
            standard_graph = []
            #for node_imp in soft_atts:
                #if (node_imp - mean(soft_atts)) != 0:
                #standard_graph.append((node_imp) / (max(soft_atts)))

                #else:
                #    standard_graph.append(0)
            standard_graph.append(softmax(soft_atts))
            standard_attributions.append(standard_graph)

        return standard_attributions

    def softmax(vector):
        e = exp(vector)
        return e / e.sum()
    def drop_important_nodes(self, graph, importance_threshold):
        square_grads_wrt_graph_label, square_grads_wrt_class_zero, square_grads_wrt_class_one = self.compute_guided_gradients(
          self.GCN_model, graph)
        #print(len(square_grads_wrt_graph_label))
        GuidedBP_attribution_scores = self.saliency(graph, square_grads_wrt_graph_label)
        occluded_GNNgraph_list = []

        for i in range(len(GuidedBP_attribution_scores)):
            sample_graph = deepcopy(graph[i])
            graph_dict = {}
            for j in range(len(sample_graph.x)):
                if self.is_salient(j, (GuidedBP_attribution_scores[i][j]), importance_threshold):
                    # print("before: ", sample_graph.x[j])
                    sample_graph.x[j][:] = 0
                    # print(torch.zeros_like(sample_graph.x[j]))
                    # print("manipulated: ",sample_graph.x[j])
                    graph_dict[j] = True
                else:
                    graph_dict[j] = False
            self.importance_dict[i] = graph_dict
            occluded_GNNgraph_list.append(sample_graph)
        Standard_GuidedBP_attribution_scores = self.standardize_by_softmax(GuidedBP_attribution_scores)
        return occluded_GNNgraph_list, GuidedBP_attribution_scores



#dataset = TUDataset(root='data/TUDataset', name='MUTAG')

#new_output = GuidedBP_GC(task="Graph Classification", method="GuidedBP", model_name="GCN_plus_GAP", graph=[dataset[0]],
#                         importance_threshold=0.5, load_index=200, input_dim=len(dataset[0].x[0]), hid_dim=7, output_dim=2,
#                         normalize_coeff=10)
#print(new_output.new_graph[-1].x, dataset[-1].x)
#print(new_output.it_took)
#print(new_output.importance_dict)
#print(new_output.saliency_maps)