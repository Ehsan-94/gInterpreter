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
import torch.nn.functional as F
from scipy.special import softmax
from statistics import mean
from numpy import exp


class LRP_GC(object):
    def __init__(self, task, method, model_name, graph, importance_threshold, load_index, input_dim, hid_dim, output_dim,
                 normalize_coeff, DataSet_name):
        self.normalize_coeff = normalize_coeff
        self.DataSet_name = DataSet_name
        self.GNN_model = self.load_model(task, method, model_name, load_index=load_index, input_dim=input_dim, hid_dim=hid_dim,
                                         output_dim=output_dim, DataSet_name=self.DataSet_name)
        self.epsilon = 1e-16
        self.criterion = torch.nn.CrossEntropyLoss()
        self.importance_dict = {}
        start_time = perf_counter()
        self.new_graph, self.saliency_maps = self.drop_important_nodes(self.GNN_model, graph, importance_threshold)
        self.it_took = perf_counter() - start_time

    def load_model(self, task, method, model_name, load_index, input_dim, hid_dim, output_dim, DataSet_name):

        if load_index != 0:
            GNN_model, optimizer, load_index = self.loading_config(task, method, model_name, load_index, input_dim, hid_dim,
                                                                   output_dim, DataSet_name)
            return GNN_model
        else:
            GNN_model = Graph_Network.GCN_plus_GAP(model_name=model_name, model_level='graph', input_dim=input_dim,
                                                   hidden_dim=hid_dim, output_dim=output_dim, num_hid_layers=2,
                                                   Bias=True, act_fun='eLu', Weight_Initializer=1, dropout_rate=0.1)
            return GNN_model

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

    def accumulate_weights(self, model_for_you):
        gconv1_weight = model_for_you.GConvs[0].lin.weight.detach().tolist()

        gconv2_weight = model_for_you.GConvs[1].lin.weight.detach().tolist()

        ffn_weight = model_for_you.ffn.weight.detach().tolist()

        return gconv1_weight, gconv2_weight, ffn_weight

    def GNN_Model_LRP(self, your_model, test_dataset):
        FFN_activations = []
        GConv2_activations = []
        GConv1_activations = []

        your_model.eval()
        for batch_of_graphs in test_dataset:
            Output_of_Hidden_Layers, pooling_layer_output, ffn_output, soft = your_model(batch_of_graphs)
            wrt = soft.argmax(dim=1).detach().tolist()[0]
            GConv1_activations.append(torch.squeeze(Output_of_Hidden_Layers[0]).tolist())
            GConv2_activations.append(torch.squeeze(Output_of_Hidden_Layers[1]).tolist())
            FFN_activations.append(torch.squeeze(ffn_output).tolist())

        return GConv1_activations, GConv2_activations, FFN_activations

    def my_relu(self, input):
        return np.maximum(0, input)

    def Compute_R_J(self, epsilon, preceding_layer_activations, exceding_layer_weights, next_layer_R_k, is_it_last):
        #print("preceding_layer_activations: ", len(preceding_layer_activations))
        #print("preceding_layer_activations[0]: ", len(preceding_layer_activations[0]))

        Denominators = []
        for i in range(len(preceding_layer_activations)):
            Denominators_Node = []
            for j in range(len(preceding_layer_activations[i])):
                SUM_Denominator = 0
                for k in range(len(exceding_layer_weights[j])):
                    SUM_Denominator = SUM_Denominator + preceding_layer_activations[i][j] * self.my_relu(
                        exceding_layer_weights[j][k])
                Denominators_Node.append(epsilon + SUM_Denominator)
            Denominators.append(Denominators_Node)
        #print("Denominator is done: ", Denominators)
        #print("Denominator Length: ", len(Denominators))

        if is_it_last:
            Numerators = []
            for i in range(len(preceding_layer_activations)):
                Node_Numerator = []
                for j in range(len(preceding_layer_activations[i])):
                    SUM_Nominator = 0
                    for k in range(len(exceding_layer_weights[j])):
                        SUM_Nominator = SUM_Nominator + preceding_layer_activations[i][j] * self.my_relu(
                            exceding_layer_weights[j][k]) * next_layer_R_k[k]
                        # print(next_layer_R_k[k])
                    Node_Numerator.append(SUM_Nominator)
                Numerators.append(Node_Numerator)
        else:
            Numerators = []
            for i in range(len(preceding_layer_activations)):
                Node_Numerator = []
                for j in range(len(preceding_layer_activations[i])):
                    SUM_Nominator = 0
                    for k in range(len(exceding_layer_weights[j])):
                        SUM_Nominator = SUM_Nominator + preceding_layer_activations[i][j] * self.my_relu(
                            exceding_layer_weights[j][k]) * next_layer_R_k[i][k]
                    Node_Numerator.append(SUM_Nominator)
                Numerators.append(Node_Numerator)

        R_Js_Graph = []
        for i in range(len(Numerators)):
            R_Js_Graph.append([sum(x) / sum(Denominators[i]) for x in Numerators])

        return R_Js_Graph

    def Compute_R_K(self, FFN_activations, wrt):
        # print(FFN_activations)
        if wrt == 2:  # .     Graph Label
            last_layer_R_k = [0] * len(FFN_activations)
            last_layer_R_k[FFN_activations.index(max(FFN_activations))] = FFN_activations[
                FFN_activations.index(max(FFN_activations))]
            # print(last_layer_R_k)
            return last_layer_R_k
        elif wrt == 0:  # .    Class 0
            last_layer_R_k = [0] * len(FFN_activations)
            last_layer_R_k[0] = FFN_activations[0]
            # print(last_layer_R_k)
            return last_layer_R_k
        elif wrt == 1:  # .    Class 1
            last_layer_R_k = [0] * len(FFN_activations)
            last_layer_R_k[1] = FFN_activations[1]
            # print(last_layer_R_k)
            return last_layer_R_k

    def One_Graph_LRP(self, epsilon, wrt, graph_sample, weights, activations):
        GConv1_weight = weights[0]
        GConv1_weight_T = weights[1]

        GConv2_weight = weights[2]
        GConv2_weight_T = weights[3]

        FFN_weight = weights[4]
        FFN_weight_T = weights[5]

        GConv1_activations = activations[0]
        GConv2_activations = activations[1]
        FFN_activations = activations[2]

        last_layer_R_k = self.Compute_R_K(FFN_activations, wrt)
        # print("R_K: ",last_layer_R_k)
        #print("np.shape(last_layer_R_k): ", np.shape(last_layer_R_k))
        R_J_hidden2 = self.Compute_R_J(epsilon, GConv2_activations, FFN_weight_T, last_layer_R_k, True)
        # print("R_J_hidden2: ", R_J_hidden2)
        # print(len(R_J_hidden2))
        #print("np.shape(R_J_hidden2): ", np.shape(R_J_hidden2))
        R_J_hidden1 = self.Compute_R_J(epsilon, GConv1_activations, GConv2_weight, R_J_hidden2, False)
        # print(R_J_hidden1)
        # print(len(R_J_hidden1))
        #print("np.shape(R_J_hidden1): ", np.shape(R_J_hidden1))
        R_J_input = self.Compute_R_J(epsilon, graph_sample, GConv1_weight, R_J_hidden1, False)
        #print("R_J_input: ", R_J_input)
        #print("np.shape(R_J_input): ", np.shape(R_J_input))

        #print(len(R_J_input[0]))

        return R_J_input

    def softmax(self, vector):
        e = exp(vector)
        return e / e.sum()

    def normalize_values(self, LRP_attribution_scores):
        Normalized_LRP_attribution_scores = []
        for graph_lrps in LRP_attribution_scores:
            new_lrps = []
            for dim in graph_lrps:
                new_lrps.append(((dim-min(graph_lrps))*self.normalize_coeff)/(max(graph_lrps)-min(graph_lrps)))
            Normalized_LRP_attribution_scores.append(new_lrps)
        return Normalized_LRP_attribution_scores

    def Normalize_LRPs(self, your_model, dataset):
        GConv1_activations, GConv2_activations, FFN_activations = self.GNN_Model_LRP(your_model, dataset)
        GConv1_weight, GConv2_weight, FFN_weight = self.accumulate_weights(your_model)

        GConv1_weight_T = np.array(GConv1_weight).transpose()
        GConv1_weight_T = GConv1_weight_T.tolist()

        GConv2_weight_T = np.array(GConv2_weight).transpose()
        GConv2_weight_T = GConv2_weight_T.tolist()

        FFN_weight_T = np.array(FFN_weight).transpose()
        FFN_weight_T = FFN_weight_T.tolist()


        LRPs_Testset = []
        for i in range(len(dataset)):
            Output_of_Hidden_Layers, pooling_layer_output, ffn_output, soft = your_model(dataset[i])
            wrt = soft.argmax(dim=1).detach().tolist()[0]
            #print("wrt: ",wrt)
            LRPs_Testset.append(self.One_Graph_LRP(self.epsilon, wrt, dataset[i].x.detach().tolist(),
                                                   [GConv1_weight, GConv1_weight_T, GConv2_weight, GConv2_weight_T,
                                                    FFN_weight, FFN_weight_T],
                                                   [GConv1_activations[i], GConv2_activations[i], FFN_activations[i]]))
        LRPs_Testset = np.sum(LRPs_Testset, axis=1)
        Normalized_LRP_attribution_scores = self.normalize_values(LRPs_Testset)
        #print("Normalized_LRP_attribution_scores: ", Normalized_LRP_attribution_scores)
        Normalized_LRPs = []
        for i in range(len(Normalized_LRP_attribution_scores)):
            graph_norms = []
            for j in range(len(Normalized_LRP_attribution_scores[i])):
                graph_norms.append(Normalized_LRP_attribution_scores[i][j]/max(Normalized_LRP_attribution_scores[i]))
            Normalized_LRPs.append(graph_norms)
            #print(Each_Graph)
            #norm = [(float(i)) / (max(Each_Graph) + 1e-16) for i in Each_Graph]
            #Normalized_LRPs.append(Each_Graph)
        #print("Normalized_LRPs: ", Normalized_LRPs)
        return Normalized_LRPs

    def is_salient(self, score, importance_threshold):
        if importance_threshold == score == 0:
            return True
        if importance_threshold == score == 1:
            return False
        if importance_threshold < score:
            return True
        else:
            return False


    def drop_important_nodes(self, your_model, your_dataset, importance_threshold):
        LRP_attribution_scores = self.Normalize_LRPs(your_model, your_dataset)
        occluded_GNNgraph_list = []

        for i in range(len(LRP_attribution_scores)):
            sample_graph = deepcopy(your_dataset[i])
            graph_dict = {}
            for j in range(len(sample_graph.x)):

                if self.is_salient(LRP_attribution_scores[i][j], importance_threshold):
                    # print("before: ", sample_graph.x[j])
                    sample_graph.x[j][:] = 0
                    # print(torch.zeros_like(sample_graph.x[j]))
                    # print("manipulated: ",sample_graph.x[j])
                    graph_dict[j] = True
                else:
                    graph_dict[j] = False
            self.importance_dict[i] = graph_dict
            occluded_GNNgraph_list.append(sample_graph)
        #Standard_LRP_attribution_scores = self.standardize_by_softmax(LRP_attribution_scores)
        return occluded_GNNgraph_list, LRP_attribution_scores



#dataset = TUDataset(root='data/TUDataset', name='MUTAG')


#new_output = LRP_GC(task="Graph Classification", method="LRP", model_name="GCN_plus_GAP", graph=[dataset[0]],
#                    importance_threshold=0.5, load_index=200, input_dim=len(dataset[0].x[0]), hid_dim=7,
#                    output_dim=2, normalize_coeff=100, DataSet_name="MUTAG")
# print(new_output.new_graph[-1].x, dataset[-1].x)
#print(new_output.saliency_maps)
#print(new_output.importance_dict)


#[[0.6877214899367055, 0.6877214899367055, 0.7112206996019833, 0.6674892867275657, 0.6474803095736253, 0.7112206996019833, 0.7112206996019833, 0.7112206996019833, 0.6474803095736253, 0.6674893509330294, 0.7112206996019833, 0.7112206996019833, 1.0, 0.7346777874194694, 0.6731117536312806, 0.8219072613478162, 0.8219072613478162]]]