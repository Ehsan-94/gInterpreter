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
import torch.nn.functional as F
from scipy.special import softmax
from statistics import mean
from numpy import exp


class CAM_GC(object):
    def __init__(self, task, method, model_name, graph, importance_threshold, load_index, input_dim, hid_dim, output_dim,
                 normalize_coeff, DataSet_name):
        self.DataSet_name = DataSet_name
        self.normalize_coeff = normalize_coeff
        self.GCN_model = self.load_model(task, method, model_name, load_index=load_index, input_dim=input_dim,
                                         hid_dim=hid_dim, output_dim=output_dim)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.importance_dict = {}
        start_time = perf_counter()
        self.new_graph, self.saliency_maps = self.drop_important_nodes(self.GCN_model, graph, importance_threshold)
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

        optimizer = torch.optim.Adam(GCN_model.parameters(), lr=0.001, weight_decay = 1e-6)
        checkpoint = torch.load(str(model_name) + " " + str(method) + " " + str(task) + " " + str(self.DataSet_name) + " " + str(load_index)+".pt")
        GCN_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return GCN_model, optimizer, epoch



    def loss_calculations(self, preds, gtruth):
        loss_per_epoch = self.criterion(preds, gtruth)
        return loss_per_epoch


    def weights_of_model(self, model):
        Dense_Layer_Weights = model.ffn.weight
        # Dense_Layer_Biases = model.lin.bias.data

        # GConv3_Layer_Weights = model.conv3.lin.weight.data
        GConv2_Layer_Weights = model.GConvs[1].lin.weight.data
        GConv1_Layer_Weights = model.GConvs[0].lin.weight.data

        Dense_Layer_Weights = Dense_Layer_Weights.detach().tolist()
        # GConv3_Layer_Weights = GConv3_Layer_Weights.detach().tolist()
        GConv2_Layer_Weights = GConv2_Layer_Weights.detach().tolist()
        GConv1_Layer_Weights = GConv1_Layer_Weights.detach().tolist()

        # return GConv1_Layer_Weights, GConv2_Layer_Weights, GConv3_Layer_Weights, Dense_Layer_Weights
        return GConv1_Layer_Weights, GConv2_Layer_Weights, Dense_Layer_Weights

    def CAM_FeatureMAP_production(self, model, your_dataset):
        index_of_winner_labels = []
        FeatureMaps_of_the_Last_Conv = []
        output_of_the_GAP = []
        Final_predictions = []

        model.eval()
        for batched_data in your_dataset:
            #print(batched_data.x)
            CAM_Test_Convs, CAM_Test_GAP, FFN_OutPut, prob = model(batched_data)
            # index_of_winner_label = CAM_output.argmax(dim=1)
            #logits = F.log_softmax(CAM_output, dim=1)
            #prob = F.softmax(logits, dim=1)
            index_of_winner_label = prob.argmax(dim=1)
            index_of_winner_labels.append(index_of_winner_label.detach().tolist())

            Final_predictions.append(prob.detach().tolist())
            output_of_the_GAP.append(CAM_Test_GAP.detach().tolist())
            FeatureMaps_of_the_Last_Conv.append(CAM_Test_Convs[1].detach().tolist())
        return FeatureMaps_of_the_Last_Conv, output_of_the_GAP, Final_predictions, index_of_winner_labels

    def weight_wrt_class_and_performance(self, index_of_winner_labels, Dense_Layer_Weights):
        Weights_of_the_Predicted_Class = []
        for i in range(len(index_of_winner_labels)):
            Weights_of_the_Predicted_Class.append(Dense_Layer_Weights[index_of_winner_labels[i][0]])

        Weights_of_the_Class_0 = []
        for i in range(len(index_of_winner_labels)):
            Weights_of_the_Class_0.append(Dense_Layer_Weights[0])

        Weights_of_the_Class_1 = []
        for i in range(len(index_of_winner_labels)):
            Weights_of_the_Class_1.append(Dense_Layer_Weights[1])

        return Weights_of_the_Predicted_Class, Weights_of_the_Class_0, Weights_of_the_Class_1

    def CAM_Attribution_Scores(self, model, your_dataset):
        GConv1_Layer_Weights, GConv2_Layer_Weights, Dense_Layer_Weights = self.weights_of_model(model)
        FeatureMaps_of_the_Last_Conv, output_of_the_GAP, Final_predictions, index_of_winner_labels = self.CAM_FeatureMAP_production(
            model, your_dataset)
        Weights_of_the_Predicted_Class, Weights_of_the_Class_0, Weights_of_the_Class_1 = self.weight_wrt_class_and_performance(
            index_of_winner_labels, Dense_Layer_Weights)
        Weights_and_Maps_Multiplication_on_Nodes_of_each_graph = []
        Normalized_Attributions = []

        for i in range(len(FeatureMaps_of_the_Last_Conv)):
            Each_Graph = []
            for j in range(len(FeatureMaps_of_the_Last_Conv[i])):
                Each_Graph.append(np.multiply(Weights_of_the_Predicted_Class[i], FeatureMaps_of_the_Last_Conv[i][j]))
            Weights_and_Maps_Multiplication_on_Nodes_of_each_graph.append(Each_Graph)

        for i in range(len(FeatureMaps_of_the_Last_Conv)):
            Each_Graph = []
            for j in range(len(FeatureMaps_of_the_Last_Conv[i])):
                Each_Graph.append(sum(Weights_and_Maps_Multiplication_on_Nodes_of_each_graph[i][j]))
            norm = [(float(i) - min(Each_Graph)) * self.normalize_coeff / (max(Each_Graph) - min(Each_Graph)) for i in Each_Graph]
            #norm = softmax(Each_Graph).tolist()
            Normalized_Attributions.append(norm)
            #print("Normalized_Attributions: ",Normalized_Attributions)
        Normalized_Attributions_final = []
        for graph in Normalized_Attributions:
            Normalized_Attributions_final.append([node/max(graph) for node in graph])
        #print("Normalized_Attributions_final: ", Normalized_Attributions_final)
        return Normalized_Attributions_final

    def softmax(vector):
        e = exp(vector)
        return e / e.sum()
    def is_salient(self, score, importance_threshold):
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
            for node_imp in soft_atts:
                #if (node_imp - mean(soft_atts)) != 0:
                standard_graph.append((node_imp) / (max(soft_atts)))
                #else:
                #    standard_graph.append(0)
            standard_attributions.append(standard_graph)

        return standard_attributions

    def drop_important_nodes(self, your_model, your_dataset, importance_threshold):
        CAM_attribution_scores = self.CAM_Attribution_Scores(your_model, your_dataset)
        occluded_GNNgraph_list = []

        for i in range(len(CAM_attribution_scores)):
            sample_graph = deepcopy(your_dataset[i])
            graph_dict = {}
            for j in range(len(sample_graph.x)):

                if self.is_salient(CAM_attribution_scores[i][j], importance_threshold):
                    # print("before: ", sample_graph.x[j])
                    sample_graph.x[j][:] = 0
                    # print(torch.zeros_like(sample_graph.x[j]))
                    # print("manipulated: ",sample_graph.x[j])
                    graph_dict[j] = True
                else:
                    graph_dict[j] = False
            self.importance_dict[i] = graph_dict
            occluded_GNNgraph_list.append(sample_graph)

        #Standard_CAM_attribution_scores = self.standardize_by_softmax(CAM_attribution_scores)
        return occluded_GNNgraph_list, CAM_attribution_scores



#dataset = TUDataset(root='data/TUDataset', name='MUTAG')

#new_output = CAM_GC(task="Graph Classification", method="CAM", model_name="GCN_plus_GAP", graph=[dataset[0]],
#                    importance_threshold=0.5, load_index=200, input_dim=len(dataset[0].x[0]), hid_dim=7, output_dim=2,
#                    normalize_coeff=10, DataSet_name="MUTAG")
#print(new_output.new_graph[-1].x, dataset[-1].x)
#print(new_output.it_took)
#print(new_output.importance_dict)
#print(new_output.saliency_maps)