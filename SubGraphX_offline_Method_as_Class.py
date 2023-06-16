import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas
from torch_geometric.datasets import TUDataset
import torch
from torch_geometric.loader import DataLoader
import GCN_plus_GAP as Graph_Network
from sklearn import metrics
import statistics
from copy import deepcopy
from torch.nn import ReLU, Sequential
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn.parameter import Parameter
import numpy as np
from time import perf_counter


class SubGraphX_off_the_fly(object):
    def __init__(self, your_dataset, Task_name, Model_Name, classifier_load_index, loading_graph_index, input_dim,
                 hid_dim, output_dim, category, DataSet_name):
        self.DataSet_name = DataSet_name
        self.Model_Name = Model_Name
        self.loading_graph_index = loading_graph_index
        Explainability_name = "SubGraphX"

        self.GNN_Model = self.load_model(Task_name=Task_name, Explainability_name=Explainability_name,
                                         Model_Name=self.Model_Name, classifier_load_index=classifier_load_index,
                                         input_dim=input_dim, hid_dim=hid_dim, output_dim=output_dim, category=category,
                                         loading_graph_index=loading_graph_index)
        self.masked_data_class0, self.maskout_data_class0, self.masked_pred_class0, self.maskout_pred_class0, self.class0_input_graph, self.masked_data_class1, self.maskout_data_class1, self.masked_pred_class1, self.maskout_pred_class1, self.class1_input_graph, self.masked_data_label, self.maskout_data_label, self.masked_pred_label, self.maskout_pred_label, self.label_input_graph, self.it_took = self.reconfig_data(your_dataset, Explainability_name, Task_name, loading_graph_index)

        self.saliency_maps, self.importance_dict = self.drop_important_nodes(your_dataset)

    def load_model(self, Task_name, Explainability_name, Model_Name, classifier_load_index, input_dim, hid_dim,
                   output_dim, category, loading_graph_index):

        if classifier_load_index != 0:
            GNN_Model, optimizer, classifier_load_index = self.loading_config(Task_name=Task_name,
                                                                              Explainability_name=Explainability_name,
                                                                              Model_Name=Model_Name,
                                                                              classifier_load_index=classifier_load_index,
                                                                              input_dim=input_dim, hid_dim=hid_dim,
                                                                              output_dim=output_dim,
                                                                              category=category,
                                                                              loading_graph_index=loading_graph_index)
            return GNN_Model
        else:
            GNN_Model = Graph_Network.GCN_plus_GAP(model_name=Model_Name, model_level='graph', input_dim=input_dim,
                                                   hidden_dim=hid_dim, output_dim=output_dim, num_hid_layers=2,
                                                   Bias=True, act_fun='eLu', Weight_Initializer=1, dropout_rate=0.1)
            return GNN_Model

    def loading_config(self, Task_name, Explainability_name, Model_Name, classifier_load_index, input_dim, hid_dim,
                       output_dim, category, loading_graph_index):
        GNN_Model = Graph_Network.GCN_plus_GAP(model_name=Model_Name, model_level='graph', input_dim=input_dim,
                                               hidden_dim=hid_dim, output_dim=output_dim, num_hid_layers=2, Bias=True,
                                               act_fun='eLu', Weight_Initializer=1, dropout_rate=0.1)
        optimizer = torch.optim.Adam(params=GNN_Model.parameters(), lr=0.001, weight_decay=1e-6)
        checkpoint = torch.load(Model_Name + " " + Explainability_name + " " + Task_name + " " + self.DataSet_name + " " + str(classifier_load_index)+".pt")
        GNN_Model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return GNN_Model, optimizer, epoch

    def reconfig_data(self, your_dataset, Explainability_name, Task_name, loading_graph_index):
        masked_data_class0_list = []
        maskout_data_class0_list = []
        masked_pred_class0_list = []
        maskout_pred_class0_list = []
        class0_input_graph = []

        masked_data_class1_list = []
        maskout_data_class1_list = []
        masked_pred_class1_list = []
        maskout_pred_class1_list = []
        class1_input_graph = []

        masked_data_label_list = []
        maskout_data_label_list = []
        masked_pred_label_list = []
        maskout_pred_label_list = []
        label_input_graph = []

        for i, graph_b in enumerate(your_dataset):
            mask_data_class0, maskout_data_class0, masked_pred_class0, maskout_pred_class0, graph0, mask_data_class1, maskout_data_class1, masked_pred_class1, maskout_pred_class1, graph1, mask_data_label, maskout_data_label, masked_pred_label, maskout_pred_label, graph2, time_for_graph_label = self.load_data(
                Explainability_name, Task_name, loading_graph_index)

            masked_data_class0_list.append(mask_data_class0)
            maskout_data_class0_list.append(maskout_data_class0)
            masked_pred_class0_list.append(masked_pred_class0)
            maskout_pred_class0_list.append(maskout_pred_class0)
            class0_input_graph.append(graph0)

            masked_data_class1_list.append(mask_data_class1)
            maskout_data_class1_list.append(maskout_data_class1)
            masked_pred_class1_list.append(masked_pred_class1)
            maskout_pred_class1_list.append(maskout_pred_class1)
            class1_input_graph.append(graph1)

            masked_data_label_list.append(mask_data_label)
            maskout_data_label_list.append(maskout_data_label)
            masked_pred_label_list.append(masked_pred_label)
            maskout_pred_label_list.append(maskout_pred_label)
            label_input_graph.append(graph2)

        return masked_data_class0_list, maskout_data_class0_list, masked_pred_class0_list, maskout_pred_class0_list, class0_input_graph, masked_data_class1_list, maskout_data_class1_list, masked_pred_class1_list, maskout_pred_class1_list, class1_input_graph, masked_data_label_list, maskout_data_label_list, masked_pred_label_list, maskout_pred_label_list, label_input_graph, time_for_graph_label

    def load_data(self, Explainability_name, Task_name, loading_graph_index):
        checkpoint_class0 = torch.load("SubGraphX_Files/" + "SubGraphX Explainer" + self.Model_Name + "_graph_" + "important_for_class_zero" + "_" + str(
                loading_graph_index) + ".pt")
        mask_data_class0 = checkpoint_class0['mask_data']
        maskout_data_class0 = checkpoint_class0['maskout_data']
        masked_pred_class0 = checkpoint_class0["masked_pred"]
        maskout_pred_class0 = checkpoint_class0["maskout_pred"]
        graph0 = checkpoint_class0["input_graph"]

        checkpoint_class1 = torch.load("SubGraphX_Files/" + "SubGraphX Explainer" + self.Model_Name + "_graph_" + "important_for_class_one" + "_" + str(
                loading_graph_index) + ".pt")
        mask_data_class1 = checkpoint_class1['mask_data']
        maskout_data_class1 = checkpoint_class1['maskout_data']
        masked_pred_class1 = checkpoint_class1["masked_pred"]
        maskout_pred_class1 = checkpoint_class1["maskout_pred"]
        graph1 = checkpoint_class1["input_graph"]

        checkpoint_associated_label = torch.load("SubGraphX_Files/" + "SubGraphX Explainer" + self.Model_Name + "_graph_" + "associated_label" + "_" + str(
                loading_graph_index) + ".pt")
        mask_data_label = checkpoint_associated_label['mask_data']
        maskout_data_label = checkpoint_associated_label['maskout_data']
        masked_pred_label = checkpoint_associated_label["masked_pred"]
        maskout_pred_label = checkpoint_associated_label["maskout_pred"]
        graph2 = checkpoint_associated_label["input_graph"]
        time_for_graph_label = checkpoint_associated_label["sample_specific_Explanation_time"]

        return mask_data_class0, maskout_data_class0, masked_pred_class0, maskout_pred_class0, graph0, mask_data_class1, maskout_data_class1, masked_pred_class1, maskout_pred_class1, graph1, mask_data_label, maskout_data_label, masked_pred_label, maskout_pred_label, graph2, time_for_graph_label

    def Compute_ROC_AUC(self, your_model, main_dataset, your_dataset, masked):
        preds = []
        reals = []
        if masked == False:
            your_model.eval()
            for batched_data in main_dataset:
                # final_GNN_layer_output, sortpooled_embedings, output_conv1d_1, maxpooled_output_conv1d_1, output_conv1d_2, to_dense, output_h1, dropout_output_h1, output_h2, softmaxed_h2 = your_model(batched_data)
                # Grad_CAM_Test_One_Before_Last_Conv, Grad_CAM_Test_Last_Conv, Grad_CAM_Test_GAP, Grad_CAM_Test_out = your_model(batched_data)
                Output_of_Hidden_Layers, pooling_layer_output, ffn_output, soft = your_model(batched_data)
                # logits = F.log_softmax(Grad_CAM_Test_out, dim=1)
                # prob = F.softmax(logits, dim=1)

                preds.append(soft.cpu().detach())

        else:

            your_model.eval()
            for i, batched_data in enumerate(your_dataset):
                # final_GNN_layer_output, sortpooled_embedings, output_conv1d_1, maxpooled_output_conv1d_1, output_conv1d_2, to_dense, output_h1, dropout_output_h1, output_h2, softmaxed_h2 = your_model(batched_data)
                # Grad_CAM_Test_One_Before_Last_Conv, Grad_CAM_Test_Last_Conv, Grad_CAM_Test_GAP, Grad_CAM_Test_out = your_model(batched_data)

                Output_of_Hidden_Layers, pooling_layer_output, ffn_output, soft = your_model(batched_data)
                # logits = F.log_softmax(Grad_CAM_Test_out, dim=1)
                # prob = F.softmax(logits, dim=1)

                preds.append(soft.cpu().detach())

        for i, batched_graph in enumerate(main_dataset):
            reals.append(batched_graph.y.tolist()[0])
            # preds = torch.cat(preds).cpu().numpy()
        # preds = preds[:, 1]
        preds = torch.cat(preds)
        # print(preds)
        preds, max_idxs = torch.max(preds[:], dim=1)

        roc_auc = metrics.roc_auc_score(reals, preds, average='macro')
        return roc_auc

    def Fidelity_computation(self, your_model, your_dataset, maskout_data_label):
        auc_roc_before_droping_important_nodes = self.Compute_ROC_AUC(your_model, your_dataset, maskout_data_label,
                                                                      False)
        print("auc_roc_before_droping_important_nodes: ", auc_roc_before_droping_important_nodes)
        auc_roc_after_droping_important_nodes = self.Compute_ROC_AUC(your_model, your_dataset, maskout_data_label, True)
        print("auc_roc_after_droping_important_nodes: ", auc_roc_after_droping_important_nodes)

        return auc_roc_before_droping_important_nodes - auc_roc_after_droping_important_nodes

    def crs_subgraph_saliency(self, your_dataset, masked_data_class0, masked_data_class1):
        binary_scores_class0 = []
        binary_scores_class1 = []
        for graph_main, graph0, graph1 in zip(your_dataset, masked_data_class0, masked_data_class1):
            binary_score_class0 = ''
            binary_score_class1 = ''
            for node_main, node0, node1 in zip(graph_main.x, graph0.x, graph1.x):
                if sum(node0) > 0:
                    binary_score_class0 += '1'
                else:
                    binary_score_class0 += '0'
                if sum(node1) > 0:
                    binary_score_class1 += '1'
                else:
                    binary_score_class1 += '0'
            binary_scores_class0.append(binary_score_class0)
            binary_scores_class1.append(binary_score_class1)
        # print("binary_scores_class0: ", binary_scores_class0)
        # print("binary_scores_class1: ", binary_scores_class1)
        return binary_scores_class0, binary_scores_class1

    def hamming_distance(self, string1, string2):

        distance = 0
        L = len(string1)
        for i in range(L):
            if string1[i] != string2[i]:
                distance += 1
        return distance

    def compute_contrastivity(self, your_dataset, masked_data_class0, masked_data_class1):
        binary_scores_class0, binary_scores_class1 = self.crs_subgraph_saliency(your_dataset, masked_data_class0,
                                                                                masked_data_class1)
        h_dist_list = []
        for cor_bin_scores, incor_bin_scores in zip(binary_scores_class0, binary_scores_class1):
            h_distance = self.hamming_distance(cor_bin_scores, incor_bin_scores) / len(cor_bin_scores)
            h_dist_list.append(h_distance)
        return statistics.mean(h_dist_list)

    def spr_subgraph_saliency(self, masked_data_class0, masked_data_class1):
        binary_scores_class0 = []
        binary_scores_class1 = []
        for graph0, graph1 in zip(masked_data_class0, masked_data_class1):
            binary_score_class0 = []
            binary_score_class1 = []
            for node0, node1 in zip(graph0.x, graph1.x):
                if sum(node0) > 0:
                    binary_score_class0.append(1)
                else:
                    binary_score_class0.append(0)
                if sum(node1) > 0:
                    binary_score_class1.append(1)
                else:
                    binary_score_class1.append(0)
            binary_scores_class0.append(binary_score_class0)
            binary_scores_class1.append(binary_score_class1)
        # print("binary_scores_class0: ", binary_scores_class0)
        # print("binary_scores_class1: ", binary_scores_class1)
        return binary_scores_class0, binary_scores_class1

    def compute_sparsity(self, masked_data_class0, masked_data_class1):
        binary_scores_class0, binary_scores_class1 = self.spr_subgraph_saliency(masked_data_class0, masked_data_class1)

        sparsity_list = []
        for cor_binary_score, incor_binary_score in zip(binary_scores_class0, binary_scores_class1):
            sparsity = 1 - ((sum(cor_binary_score) + sum(incor_binary_score)) / (2 * len(incor_binary_score)))
            sparsity_list.append(sparsity)

        return statistics.mean(sparsity_list)

    def drop_important_nodes(self, your_dataset):
        #fid_score = self.Fidelity_computation(self.GNN_Model, your_dataset, self.maskout_data_label)
        # print("Fidelity: ", fid_score)

        #crs_score = self.compute_contrastivity(your_dataset, self.masked_data_class0, self.masked_data_class1)
        # print("Contrastivity: ", crs_score)

        #spr_score = self.compute_sparsity(self.masked_data_class0, self.masked_data_class1)
        # print("Sparsity: ", spr_score)

        saliency_maps = {}
        importance_dict = {}
        for i in range(len(self.masked_data_label)):
            saliency_maps[i] = {}
            importance_dict[i] = {}
            for j, node_feats in enumerate(self.masked_data_label[i].x):
                if torch.sum(node_feats) == 1:
                    saliency_maps[i][j] = True
                    importance_dict[i][j] = True
                else:
                    saliency_maps[i][j] = False
                    importance_dict[i][j] = False

        #print("saliency_maps: ", saliency_maps)
        return saliency_maps, importance_dict

#dataset = TUDataset(root='data/TUDataset', name='MUTAG')
#subgx_out_the_fly = SubGraphX_off_the_fly(your_dataset=dataset, Task_name='Graph Classification',
#                                          Model_Name="GCN_plus_GAP", classifier_load_index=200,
#                                          input_dim=7, hid_dim=7, output_dim=2, loading_graph_index=1,
#                                          category='correct', DataSet_name="MUTAG")

#print(subgx_out_the_fly.it_took, subgx_out_the_fly.saliency_maps, subgx_out_the_fly.importance_dict)
