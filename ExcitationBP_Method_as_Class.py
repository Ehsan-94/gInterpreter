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

class ExcitationBP_GC(object):
    def __init__(self, task, method, model_name, graph, importance_threshold, load_index, input_dim, hid_dim, output_dim,
                 normalize_coeff, DataSet_name):
        self.normalize_coeff = normalize_coeff
        self.DataSet_name = DataSet_name
        self.GNN_model = self.load_model(task, method, model_name, load_index=load_index, input_dim=input_dim, hid_dim=hid_dim,
                                         output_dim=output_dim)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.importance_dict = {}
        self.new_graph, self.saliency_maps = self.drop_important_nodes(self.GNN_model, graph, importance_threshold)

    def load_model(self, task, method, model_name, load_index, input_dim, hid_dim, output_dim):

        if load_index != 0:
            GNN_model, optimizer, load_index = self.loading_config(task, method, model_name, load_index, input_dim, hid_dim,
                                                                   output_dim)
            return GNN_model
        else:
            GNN_model = Graph_Network.GCN_plus_GAP(model_name=model_name, model_level='graph', input_dim=input_dim,
                                                   hidden_dim=hid_dim, output_dim=output_dim, num_hid_layers=2,
                                                   Bias=True, act_fun='eLu', Weight_Initializer=1, dropout_rate=0.1)
            return GNN_model

    def loading_config(self, task, method, model_name, load_index, input_dim, hid_dim, output_dim):
        GNN_model = Graph_Network.GCN_plus_GAP(model_name=model_name, model_level='graph', input_dim=input_dim, hidden_dim=hid_dim, output_dim=output_dim, num_hid_layers=2, Bias=True, act_fun='eLu', Weight_Initializer=1, dropout_rate=0.1)
        DataSet_name = "MUTAG"
        optimizer = torch.optim.Adam(GNN_model.parameters(), lr=0.001, weight_decay = 1e-6)
        checkpoint = torch.load(str(model_name) + " " + str(method) + " " + str(task) + " " + str(DataSet_name) + " " + str(load_index)+".pt")
        GNN_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return GNN_model, optimizer, epoch

    def loss_calculations(self, preds, gtruth):
        loss_per_epoch = self.criterion(preds, gtruth)
        return loss_per_epoch

    def my_relu(self, input):
        return np.maximum(0, input)

    def accumulate_weights(self, model_for_you):
        gconv1_weight = model_for_you.GConvs[0].lin.weight.detach().tolist()

        gconv2_weight = model_for_you.GConvs[1].lin.weight.detach().tolist()

        ffn_weight = model_for_you.ffn.weight.detach().tolist()

        return gconv1_weight, gconv2_weight, ffn_weight

    def Division_by_Zero(self, epsilon, act_hat):
        for i in range(len(act_hat)):
            for j in range(len(act_hat[i])):
                if act_hat[i][j] == 0:
                    act_hat[i][j] = act_hat[i][j] + epsilon
                else:
                    act_hat[i][j] = act_hat[i][j]
        return act_hat

    def Compute_Pobabilities(self, last_layer, epsilon, preceding_layer_activations, exceding_layer_weights,
                             exceding_layer_prob):

        # 1 Weights and Activations
        weights_and_activations_Graph = []
        for i in range(len(exceding_layer_weights)):
            weights_and_activations_Node = []
            for j in range(len(preceding_layer_activations)):
                weights_and_activations_Node.append(
                    sum(np.multiply(self.my_relu(exceding_layer_weights[i]), preceding_layer_activations[j])))
            weights_and_activations_Graph.append(weights_and_activations_Node)
        # print(np.shape(weights_and_activations_Graph), weights_and_activations_Graph)
        # print("1 Multiplication of Weights and Activations: ", np.shape(weights_and_activations_Graph))
        weights_and_activations_Graph = self.Division_by_Zero(epsilon, weights_and_activations_Graph)
        weights_and_activations_Graph = np.array(weights_and_activations_Graph).transpose()
        weights_and_activations_Graph = weights_and_activations_Graph.tolist()
        # print("mul: ",np.shape(weights_and_activations_Graph), weights_and_activations_Graph)
        # 2 Point-Wise division
        # print(exceding_layer_prob)
        if last_layer:
            division_result_Graph = []
            for i in range(len(exceding_layer_prob)):
                division_result = [float(exceding_layer_prob[i] * weights_and_activations_Graph[j][i]) for j in
                                   range(len(weights_and_activations_Graph))]
                division_result_Graph.append(division_result)
            # print("2 Division Results: ", np.shape(division_result_Graph))
            division_result_Graph = np.array(division_result_Graph).transpose()
            division_result_Graph = division_result_Graph.tolist()
            # print(np.shape(division_result_Graph), division_result_Graph)
        else:
            division_result_Graph = []
            for i in range(len(exceding_layer_prob)):
                division_result = [float(exceding_layer_prob[i][j] * weights_and_activations_Graph[i][j]) for j in
                                   range(len(weights_and_activations_Graph[i]))]
                division_result_Graph.append(division_result)
            # print("2 Division Results: ", np.shape(division_result_Graph))
            # print(np.shape(division_result_Graph), division_result_Graph)

        # 3 Multiplication by Weights
        exceding_layer_weights = np.array(exceding_layer_weights).transpose()
        exceding_layer_weights = exceding_layer_weights.tolist()

        weights_third_step_Graph = []
        for i in range(len(division_result_Graph)):
            weights_third_step = []
            for j in range(len(exceding_layer_weights)):
                weights_third_step.append(
                    sum([x * y for x, y in zip(self.my_relu(exceding_layer_weights[j]), division_result_Graph[i])]))
            weights_third_step_Graph.append(weights_third_step)
        # print("3 Third Step: ", np.shape(weights_third_step_Graph))
        # print(weights_third_step_Graph)

        # 4 Forth Step
        final_probability_Graph = []
        for i in range(len(preceding_layer_activations)):
            final_probability_vector = [preceding_layer_activations[i][j] * weights_third_step_Graph[i][j] for j in
                                        range(len(preceding_layer_activations[i]))]
            final_probability_Graph.append(final_probability_vector)
        # print(np.shape(final_probability_Graph))
        # print(final_probability_Graph)

        return final_probability_Graph

    def is_salient(self, score, importance_threshold):
        if importance_threshold == score == 0:
            return True
        if importance_threshold == score == 1:
            return False
        if importance_threshold < score:
            return True
        else:
            return False



    def Compute_P_last_layer(self, FFN_activations, wrt):
        #prob = self.normalize_labels_to_probabilistics(FFN_activations)
        prob = FFN_activations
        if wrt == 2:
            last_layer_R_k = [0] * len(prob)
            last_layer_R_k[prob.index(max(prob))] = max(prob)  # 1
            return last_layer_R_k

        elif wrt == 1:
            last_layer_R_k = [0] * len(prob)
            last_layer_R_k[1] = 1  # prob[1]
            return last_layer_R_k

        elif wrt == 0:
            last_layer_R_k = [0] * len(prob)
            last_layer_R_k[0] = 1  # prob[0]
            return last_layer_R_k

    def One_Graph_EB(self, wrt, epsilon, input_sample, weights, activations):
        GConv1_weight = weights[0]
        GConv1_weight_T = weights[1]

        GConv2_weight = weights[2]
        GConv2_weight_T = weights[3]

        FFN_weight = weights[4]
        FFN_weight_T = weights[5]

        GConv1_activations = activations[0]
        GConv2_activations = activations[1]
        FFN_activations = activations[2]

        # print(FFN_activations)
        last_layer_prob = self.Compute_P_last_layer(FFN_activations, wrt)
        # print("FFN: ", last_layer_prob)

        hidden2_probability_vector = self.Compute_Pobabilities(True, epsilon, GConv2_activations, FFN_weight,
                                                               last_layer_prob)
        # print("Second GCN Probability: ",hidden2_probability_vector)
        # print("hid2: ", np.shape(hidden2_probability_vector))

        hidden1_probability_vector = self.Compute_Pobabilities(False, epsilon, GConv1_activations, GConv2_weight_T,
                                                               hidden2_probability_vector)
        # print("First GCN Probability: ",hidden2_probability_vector)
        # print("hid1: ", np.shape(hidden1_probability_vector))

        input_probability_vector = self.Compute_Pobabilities(False, epsilon, input_sample, GConv1_weight_T,
                                                             hidden1_probability_vector)
        # print("Input Probability: ",input_probability_vector)
        # print("input: ", np.shape(input_probability_vector))

        return input_probability_vector

    def transpose_weights(self, GConv1_weight, GConv2_weight, FFN_weight):
        GConv1_weight_T = np.array(GConv1_weight).transpose()
        GConv1_weight_T = GConv1_weight_T.tolist()

        GConv2_weight_T = np.array(GConv2_weight).transpose()
        GConv2_weight_T = GConv2_weight_T.tolist()

        FFN_weight_T = np.array(FFN_weight).transpose()
        FFN_weight_T = FFN_weight_T.tolist()

        return GConv1_weight_T, GConv2_weight_T, FFN_weight_T

    def softmax(vector):
        e = exp(vector)
        return e / e.sum()

    def Get_ExcitationBPs(self, your_model, dataset):
        GConv1_weight, GConv2_weight, FFN_weight = self.accumulate_weights(your_model)
        GConv1_weight_T, GConv2_weight_T, FFN_weight_T = self.transpose_weights(GConv1_weight, GConv2_weight,
                                                                                FFN_weight)

        GConv1_weight = GConv1_weight.copy()
        GConv2_weight = GConv2_weight.copy()
        FFN_weight = FFN_weight.copy()

        EBs_Testset = []
        epsilon = 1e-16
        # wrt = 0
        for i, graph in enumerate(dataset):
            Output_of_Hidden_Layers, pooling_layer_output, ffn_output, soft = your_model(graph)
            wrt = soft.argmax(dim=1).detach().tolist()[0]

            GCN_Model_test_activations = torch.squeeze(ffn_output).tolist()
            post_conv2_test_activations = torch.squeeze(Output_of_Hidden_Layers[1]).tolist()
            post_conv1_test_activations = torch.squeeze(Output_of_Hidden_Layers[0]).tolist()

            EBs_Graph = self.One_Graph_EB(wrt, epsilon, dataset[i].x.detach().tolist(),
                                          [GConv1_weight, GConv1_weight_T, GConv2_weight, GConv2_weight_T, FFN_weight,
                                           FFN_weight_T], [post_conv1_test_activations, post_conv2_test_activations,
                                                           GCN_Model_test_activations])
            graphs = []
            #print("EBs_Graph: ", np.shape(EBs_Graph))
            print("EBs_Graph: ", EBs_Graph)
            for j in range(len(EBs_Graph)):
                graphs.append(sum((EBs_Graph[j])))
            norm = [abs((float(i)*self.normalize_coeff) / (max(graphs) - min(graphs))) for i in graphs]
            EBs_Testset.append(norm)
        #print("EBs_Testset: ", EBs_Testset)
        return EBs_Testset

    def standardize_values(self, saliencies):
        standard_attributions = []
        for soft_atts in saliencies:
            standard_graph = []
            for node_imp in soft_atts:
                standard_graph.append((node_imp) / (max(soft_atts)))
        standard_attributions.append(standard_graph)

        return standard_attributions

    def drop_important_nodes(self, your_model, your_dataset, importance_threshold):
        EBP_attribution_scores = self.Get_ExcitationBPs(your_model, your_dataset)
        occluded_GNNgraph_list = []
        Standard_EBP_attribution_scores = self.standardize_values(EBP_attribution_scores)
        for i in range(len(Standard_EBP_attribution_scores)):
            sample_graph = deepcopy(your_dataset[i])
            graph_dict = {}
            for j in range(len(sample_graph.x)):

                if self.is_salient(Standard_EBP_attribution_scores[i][j], importance_threshold):
                    # print("before: ", sample_graph.x[j])
                    sample_graph.x[j][:] = 0
                    # print(torch.zeros_like(sample_graph.x[j]))
                    # print("manipulated: ",sample_graph.x[j])
                    graph_dict[j] = True
                else:
                    graph_dict[j] = False
            self.importance_dict[i] = graph_dict
            occluded_GNNgraph_list.append(sample_graph)

        return occluded_GNNgraph_list, Standard_EBP_attribution_scores#EBP_attribution_scores#Standard_EBP_attribution_scores


#dataset = TUDataset(root='data/TUDataset', name='MUTAG')

#new_output = ExcitationBP_GC(task="Graph Classification", method="ExcitationBP", model_name="GCN_plus_GAP", graph=[dataset[0]],
#                             importance_threshold=0.5, load_index=200, input_dim=len(dataset[0].x[0]), hid_dim=7,
#                             output_dim=2, normalize_coeff=100, DataSet_name="MUTAG")
#print(new_output.new_graph[-1].x, dataset[-1].x)
#print(new_output.saliency_maps)
#print(new_output.importance_dict)
