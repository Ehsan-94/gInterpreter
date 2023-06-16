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


class Grad_CAM_GC(object):
    def __init__(self, task, method, model_name, graph, importance_threshold, load_index, input_dim, hid_dim, output_dim,
                 normalize_coeff, DataSet_name):
        self.normalize_coeff = normalize_coeff
        self.DataSet_name = DataSet_name
        self.GCN_model = self.load_model(task, method, model_name, load_index=load_index, input_dim=input_dim,
                                         hid_dim=hid_dim, output_dim=output_dim, DataSet_name=self.DataSet_name)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.importance_dict = {}
        start_time = perf_counter()
        self.new_graph, self.saliency_maps = self.drop_important_nodes(self.GCN_model, graph, importance_threshold)
        self.it_took = perf_counter()-start_time

    def load_model(self, task, method, model_name, load_index, input_dim, hid_dim, output_dim, DataSet_name):

        if load_index != 0:
            GCN_model, optimizer, load_index = self.loading_config(task, method, model_name, load_index, input_dim,
                                                                   hid_dim, output_dim, DataSet_name)
            return GCN_model
        else:
            GCN_model = Graph_Network.GCN_plus_GAP(model_name=model_name, model_level='graph', input_dim=input_dim,
                                                   hidden_dim=hid_dim, output_dim=output_dim, num_hid_layers=2, Bias=True,
                                                   act_fun='eLu', Weight_Initializer=1, dropout_rate=0.1)
            return GCN_model

    def loading_config(self, task, method, model_name, load_index, input_dim, hid_dim, output_dim, DataSet_name):

        GCN_model = Graph_Network.GCN_plus_GAP(model_name=model_name, model_level='graph', input_dim=input_dim,
                                               hidden_dim=hid_dim, output_dim=output_dim, num_hid_layers=2, Bias=True,
                                               act_fun='eLu', Weight_Initializer=1, dropout_rate=0.1)
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


    def compute_grad(self, model, graph, with_respect):
        Grad_CAM_Test_One_Before_Last_Conv, Grad_CAM_Test_Last_Conv, Grad_CAM_Test_GAP, Grad_CAM_Test_out = model(
            graph)
        # print(prediction)
        if with_respect == 1:
            loss = self.loss_calculations(Grad_CAM_Test_out, graph.y)
            # print(loss)
            # print("done")
        elif with_respect == 2:
            loss = self.loss_calculations(Grad_CAM_Test_out, torch.tensor([0]))
            # print(loss)
        elif with_respect == 3:
            loss = self.loss_calculations(Grad_CAM_Test_out, torch.tensor([1]))
            # print(loss)
        return torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)


    def remove_nones(self, sample_grads):
        # print(type(sample_grads[0]))
        sample_grads2 = []
        for item in sample_grads:
            Each_Graph = []
            for item2 in item:
                if item2 != None:
                    Each_Graph.append(torch.tensor(item2.clone().detach().requires_grad_(True), requires_grad=True))
                else:
                    Each_Graph.append(torch.tensor(0))
            sample_grads2.append(Each_Graph)
            # print("separate                         ")
            # item2 = torch.tensor([0])
        # print(np.shape(sample_grads[0]))
        # print(np.shape(sample_grads2[0]))
        return sample_grads2

    def compute_sample_grads(self, model, test_dataset, with_respect):

        sample_grads = [self.compute_grad(model, graph, with_respect) for graph in test_dataset]
        # print(np.shape(sample_grads[0]))
        # print(sample_grads[20])
        sample_grads = self.remove_nones(sample_grads)
        sample_grads = zip(*sample_grads)
        sample_grads = [torch.stack(shards) for shards in sample_grads]
        # sample_grads = [print(shards) for shards in sample_grads]
        return sample_grads

    def compute_grad_cam_gradients(self, your_model, dataset):
        per_sample_grads_wrt_graph_label = self.compute_sample_grads(your_model, dataset, 1)
        per_sample_grads_wrt_class_zero = self.compute_sample_grads(your_model, dataset, 2)
        per_sample_grads_wrt_class_one = self.compute_sample_grads(your_model, dataset, 3)

        grads_wrt_graph_label = per_sample_grads_wrt_graph_label[1].detach().tolist()
        grads_wrt_class_zero = per_sample_grads_wrt_class_zero[1].detach().tolist()
        grads_wrt_class_one = per_sample_grads_wrt_class_one[1].detach().tolist()

        return grads_wrt_graph_label, grads_wrt_class_zero, grads_wrt_class_one

    def Grad_CAM_FeatureMAP_production(self, your_model, test_loader):
        index_of_winner_labels = []
        FeatureMaps_of_the_Last_Conv = []
        output_of_the_GAP = []
        Final_predictions = []

        your_model.eval()
        for batched_data in test_loader:
            Grad_CAM_Test_Convs, Grad_CAM_Test_GAP, FFN_OutPut, prob = your_model(
                batched_data)
            # index_of_winner_label = CAM_output.argmax(dim=1)
            #logits = F.log_softmax(Grad_CAM_Test_out, dim=1)
            #prob = F.softmax(logits, dim=1)
            index_of_winner_label = prob.argmax(dim=1)

            index_of_winner_labels.append(index_of_winner_label.detach().tolist())
            Final_predictions.append(prob.detach().tolist())

            FeatureMaps_of_the_Last_Conv.append(Grad_CAM_Test_Convs[1].detach().tolist())
        return FeatureMaps_of_the_Last_Conv, Final_predictions, index_of_winner_labels

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

    def column_wise_addups(self, gradients):
        return abs(np.sum(gradients, axis=1)).tolist()
    def Grad_CAM_Attribution_Scores(self, your_model, your_dataset):
        grads_wrt_graph_label, grads_wrt_class_zero, grads_wrt_class_one = self.compute_grad_cam_gradients(your_model,
                                                                                                           your_dataset)
        FeatureMaps_of_the_Last_Conv, Final_predictions, index_of_winner_labels = self.Grad_CAM_FeatureMAP_production(
            your_model, your_dataset)
        grads_wrt_graph_label = self.column_wise_addups(grads_wrt_graph_label)

        Graphs_new_gradients = []
        for graph_grads in grads_wrt_graph_label:
            new_gradients = []
            for dim in graph_grads:
                new_gradients.append(((dim-min(graph_grads))*self.normalize_coeff)/(max(graph_grads)-min(graph_grads)))
            Graphs_new_gradients.append(new_gradients)

        Grads_and_Maps_Multiplication_on_Nodes_of_each_graph = []
        Normalized_Attributions = []
        for i in range(len(FeatureMaps_of_the_Last_Conv)):
            Each_Graph = []
            for j in range(len(FeatureMaps_of_the_Last_Conv[i])):
                Each_Graph.append(sum(np.multiply(Graphs_new_gradients[i], FeatureMaps_of_the_Last_Conv[i][j])))
            Grads_and_Maps_Multiplication_on_Nodes_of_each_graph.append(Each_Graph)
        #print("Grads_and_Maps_Multiplication_on_Nodes_of_each_graph[0]: ", Grads_and_Maps_Multiplication_on_Nodes_of_each_graph[0])


        for i in range(len(Grads_and_Maps_Multiplication_on_Nodes_of_each_graph)):
            Each_Graph = []
            for j in range(len(Grads_and_Maps_Multiplication_on_Nodes_of_each_graph[i])):
                Each_Graph.append(Grads_and_Maps_Multiplication_on_Nodes_of_each_graph[i][j]/(max(Grads_and_Maps_Multiplication_on_Nodes_of_each_graph[i])))
            #norm = softmax(Each_Graph).tolist()
            Normalized_Attributions.append(Each_Graph)
        #print("Normalized_Attributions: ", Normalized_Attributions)
        return Normalized_Attributions

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
        Grad_CAM_attribution_scores = self.Grad_CAM_Attribution_Scores(your_model, your_dataset)
        occluded_GNNgraph_list = []

        for i in range(len(Grad_CAM_attribution_scores)):
            sample_graph = deepcopy(your_dataset[i])
            graph_dict = {}
            for j in range(len(sample_graph.x)):

                if self.is_salient((Grad_CAM_attribution_scores[i][j]), importance_threshold):
                    # print("before: ", sample_graph.x[j])
                    sample_graph.x[j][:] = 0
                    # print(torch.zeros_like(sample_graph.x[j]))
                    # print("manipulated: ",sample_graph.x[j])
                    graph_dict[j] = True
                else:
                    graph_dict[j] = False
            self.importance_dict[i] = graph_dict
            occluded_GNNgraph_list.append(sample_graph)
        #Standard_Grad_CAM_attribution_scores = self.standardize_by_softmax(Grad_CAM_attribution_scores)
        return occluded_GNNgraph_list, Grad_CAM_attribution_scores



#dataset = TUDataset(root='data/TUDataset', name='MUTAG')

#new_output = Grad_CAM_GC(task="Graph Classification", method="Grad-CAM", model_name="GCN_plus_GAP", graph=[dataset[0]], importance_threshold=0.5,
#                         load_index=200, input_dim=len(dataset[0].x[0]), hid_dim=7, output_dim=2, normalize_coeff=100, DataSet_name="MUTAG")
#print(new_output.new_graph[-1].x, dataset[-1].x)
#print(new_output.it_took)
#print(new_output.importance_dict)
#print(new_output.saliency_maps)