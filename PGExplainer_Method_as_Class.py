import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas
from torch_geometric.datasets import TUDataset
import torch
from torch_geometric.loader import DataLoader
import GCN_plus_GAP as Graph_Network
from copy import deepcopy
from torch.nn import ReLU, Sequential
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn.parameter import Parameter
import numpy as np
from time import perf_counter
from scipy.special import softmax
from statistics import mean


class PGExplainer(object):
    coeffs = {'edge_size': 0.05,
              'edge_ent': 1.0,
              'temp': [5.0, 2.0],
              'bias': 0.0,
              }

    def __init__(self, Model_Name, Explainability_name, Task_name,
                  classifier_load_index, explainer_save_index, Exp_Epoch, classifier_save_index,
                  Exp_lr, input_dim, hid_dim, output_dim, importance_threshold,
                  ExTrain_or_ExTest, Exp_Load_index, your_dataset, target_class,
                  DataSet_name):
        # self.GNN_Model = GNN_Model

        self.explainer_epochs = Exp_Epoch
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.importance_threshold = importance_threshold
        self.criterion = F.binary_cross_entropy_with_logits
        self.explainer_save_index = explainer_save_index
        self.Explainability_name = Explainability_name
        self.Task_name = Task_name
        self.Model_Name = Model_Name
        self.DataSet_name = DataSet_name
        self.classifier_save_index = classifier_save_index
        self.classifier_load_index = classifier_load_index
        self.input_dim = input_dim
        self.explainer_lr = Exp_lr
        self.GNN_Model = self.load_model(Task_name=Task_name, Explainability_name=Explainability_name,
                                         Model_Name=Model_Name, classifier_load_index=classifier_load_index,
                                         input_dim=input_dim, hid_dim=hid_dim, output_dim=output_dim)

        self.pgexp_mlp = Sequential(Linear(self.input_dim * 2, 64), ReLU(), Linear(64, 1))
        self.pgexp_mlp_optimizer = torch.optim.Adam(self.pgexp_mlp.parameters(), lr=self.explainer_lr)

        self.it_took, self.saliency_maps, self.importance_dict = self.drop_important_nodes(ExTrain_or_ExTest, Exp_Load_index, your_dataset,
                                                                     target_class)

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
        DataSet_name = "MUTAG"
        GNN_Model = Graph_Network.GCN_plus_GAP(model_name=Model_Name, model_level='graph', input_dim=input_dim,
                                               hidden_dim=hid_dim, output_dim=output_dim, num_hid_layers=2, Bias=True,
                                               act_fun='eLu', Weight_Initializer=1, dropout_rate=0.1)
        optimizer = torch.optim.Adam(params=GNN_Model.parameters(), lr=0.001, weight_decay=1e-6)
        checkpoint = torch.load(str(Model_Name) + " " + str(self.Explainability_name) + " " + str(self.Task_name) + " " + str(self.DataSet_name) + " " + str(self.classifier_load_index) + ".pt")
        GNN_Model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return GNN_Model, optimizer, epoch

    def explainer_loss(self, By_Perturbation_predicted_label, predicted_label):
        loss_per_epoch = self.criterion(By_Perturbation_predicted_label, predicted_label)
        return loss_per_epoch

    def train_step_explainer(self, merged_embeddings_list_batchs, GNN_Model, your_dataset, temperature,
                             GNN_Model_preds_NOT_MASKED, target_class):

        self.pgexp_mlp.train()
        self.pgexp_mlp.zero_grad()
        for batched_merged_embeddings, batched_preds_NOT_MASKED, batched_graphs in zip(merged_embeddings_list_batchs,
                                                                                       GNN_Model_preds_NOT_MASKED,
                                                                                       your_dataset):
            explaier_outputs = self.pgexp_mlp(batched_merged_embeddings).view(-1)

            edge_mask = self.binary_concrete(explaier_outputs, temperature)

            # print(edge_mask.size(), len(batched_graphs.edge_index[0]))
            self.apply_masks(GNN_Model, edge_mask, batched_graphs.edge_index, apply_sigmoid=True)

            Output_of_Hidden_Layers_MASKED, pooling_layer_output_MASKED, ffn_output_MASKED, soft_MASKED = GNN_Model(
                batched_graphs)
            # print("batched_preds_NOT_MASKED: ", batched_preds_NOT_MASKED, "   ", "batched_preds_NOT_MASKED.argmax(dim=1): ", batched_preds_NOT_MASKED.argmax(dim=1))
            if target_class == "correct":
                batch_loss = self.explainer_loss(soft_MASKED.argmax(dim=1).to(torch.float32),
                                                 batched_preds_NOT_MASKED.argmax(dim=1).to(torch.float32))
            else:
                batch_loss = self.explainer_loss(soft_MASKED.argmin(dim=1).to(torch.float32),
                                                 batched_preds_NOT_MASKED.argmin(dim=1).to(torch.float32))
            batch_loss.requires_grad = True
            batch_loss.backward(retain_graph=True)
            self.pgexp_mlp_optimizer.step()

        return edge_mask

    def train_explainer(self, GNN_Model, your_dataset, target_class):
        edge_masks_per_epoch = []
        # self.clear_masks(GNN_Model)
        merged_embeddings_list = self.get_merged_embeddings(GNN_Model, your_dataset)
        GNN_Model_preds_NOT_MASKED = []
        for batch_of_graphs in your_dataset:
            Output_of_Hidden_Layers_NOT_MASKED, pooling_layer_output_NOT_MASKED, ffn_output_NOT_MASKED, soft_NOT_MASKED = GNN_Model(
                batch_of_graphs)
            GNN_Model_preds_NOT_MASKED.append(soft_NOT_MASKED)

        for epoch in range(self.explainer_epochs):
            print("Epoch: ", epoch)
            temperature = self.compute_temp(epoch)
            # print("temperature: ", temperature)
            edge_mask = self.train_step_explainer(merged_embeddings_list, GNN_Model, your_dataset, temperature,
                                                  GNN_Model_preds_NOT_MASKED, target_class)
            edge_masks_per_epoch.append(edge_mask)

            if (epoch + 1) == self.explainer_save_index:
                torch.save({'epoch': epoch + 1, 'model_state_dict': self.pgexp_mlp.state_dict(),
                            'optimizer_state_dict': self.pgexp_mlp_optimizer.state_dict()},
                           "/content/drive/My Drive/Explainability Methods/" + str(
                               self.Explainability_name) + " on " + str(self.Task_name) + "/Model/" + str(self.Model_Name) + " " + str(self.Explainability_name) + " " + str(self.Task_name) + " " + str(self.DataSet_name) + " " + str(self.classifier_save_index) + ".pt")
        self.clear_masks(GNN_Model)

    def test_explainer(self, GNN_Model, your_dataset, pgexp_mlp):
        predicted_labels_MASKED = []
        edge_masks = []
        merged_embeddings_list_batchs = self.get_merged_embeddings(GNN_Model, your_dataset)
        GNN_Model_preds_NOT_MASKED = []
        pgexp_mlp.eval()
        for batch_of_graphs in your_dataset:
            Output_of_Hidden_Layers_NOT_MASKED, pooling_layer_output_NOT_MASKED, ffn_output_NOT_MASKED, soft_NOT_MASKED = GNN_Model(
                batch_of_graphs)
            GNN_Model_preds_NOT_MASKED.append(soft_NOT_MASKED)
        for batched_merged_embeddings, batched_preds_NOT_MASKED, batched_graphs in zip(merged_embeddings_list_batchs,
                                                                                       GNN_Model_preds_NOT_MASKED,
                                                                                       your_dataset):
            explaier_outputs = pgexp_mlp(batched_merged_embeddings).view(-1)

            temperature = 1
            edge_mask = self.binary_concrete(explaier_outputs, temperature)
            self.apply_masks(GNN_Model, edge_mask, batched_graphs.edge_index, apply_sigmoid=True)
            edge_masks.append(edge_mask)

            Output_of_Hidden_Layers_MASKED, pooling_layer_output_MASKED, ffn_output_MASKED, soft_MASKED = GNN_Model(
                batched_graphs)
            # print(soft_MASKED.argmax(dim=1))
            predicted_labels_MASKED.append(torch.squeeze(soft_MASKED.argmax(dim=1)).tolist())
        self.clear_masks(GNN_Model)
        return predicted_labels_MASKED, edge_masks

    def apply_masks(self, model, mask, edge_index, apply_sigmoid):
        loop_mask = edge_index[0] != edge_index[1]

        for module in model.modules():
            if isinstance(module, MessagePassing):

                if (not isinstance(mask, Parameter)
                        and '_edge_mask' in module._parameters):
                    mask = Parameter(mask)

                module.explain = True
                module._edge_mask = mask
                module._loop_mask = loop_mask
                module._apply_sigmoid = apply_sigmoid
                # print(module._edge_mask)

    def clear_masks(self, model):

        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.explain = False
                module._edge_mask = None
                module._loop_mask = None
                module._apply_sigmoid = True
        return module

    def get_merged_embeddings(self, GNN_Model, your_dataset):
        new_embeddings_list = []
        merged_embeddings_list = []
        for batched_graphs in your_dataset:
            new_graph_by_masks = deepcopy(batched_graphs.detach())

            new_embeddings, GNN_Model_explain_predicted_labels = self.get_hopped_embeddings(GNN_Model,
                                                                                            new_graph_by_masks)
            new_embeddings_list.append(new_embeddings)

            merged_embeddings = self.edge_embeddings(new_embeddings, new_graph_by_masks.edge_index)
            merged_embeddings_list.append(merged_embeddings)

        return merged_embeddings_list

    def edge_embeddings(self, embedding, edge_index):
        merged_embeds_dup = [embedding[edge_index[0]], embedding[edge_index[1]]]
        merged_embeds_dup = torch.cat(merged_embeds_dup, dim=-1)

        return merged_embeds_dup

    def get_hopped_embeddings(self, GNN_Model, batch_of_graphs):
        GNN_Model.eval()
        Output_of_Hidden_Layers, pooling_layer_output, ffn_output, soft = GNN_Model(batch_of_graphs)
        GNN_Model_test_pred = soft.argmax(dim=1)
        return Output_of_Hidden_Layers[-1], GNN_Model_test_pred

    def compute_temp(self, epoch):
        temp = self.coeffs['temp']
        return temp[0] * pow(temp[1] / temp[0], epoch / self.explainer_epochs)

    def binary_concrete(self, explaier_outputs, temperature):
        bias = self.coeffs['bias']
        eps = (1 - 2 * bias) * torch.rand_like(explaier_outputs) + bias
        return torch.sigmoid((eps.log() - (1 - eps).log() + explaier_outputs) / temperature)

    def load_explainer_mlp(self, Exp_Load_index, target_class):
        pgexp_mlp = Sequential(Linear(self.input_dim * 2, 64), ReLU(), Linear(64, 1))
        pgexp_mlp_optimizer = torch.optim.Adam(self.pgexp_mlp.parameters(), lr=self.explainer_lr)
        checkpoint = torch.load(str(self.Model_Name) + "_Model_classifier_PGExplainer_MLP_" + str(Exp_Load_index)+"_epochs_" + str(target_class) + ".pt")
        pgexp_mlp.load_state_dict(checkpoint['model_state_dict'])
        pgexp_mlp_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return pgexp_mlp, pgexp_mlp_optimizer

    def drop_important_nodes(self, ExTrain_or_ExTest, Exp_Load_index, your_dataset, target_class):
        if ExTrain_or_ExTest == "train":
            self.train_explainer(self.GNN_Model, your_dataset, target_class)
        elif ExTrain_or_ExTest == "test":
            pgexp_mlp, pgexp_mlp_optimizer = self.load_explainer_mlp(Exp_Load_index=Exp_Load_index,
                                                                     target_class="correct")
            start_time = perf_counter()
            predicted_labels, edge_masks = self.test_explainer(self.GNN_Model, your_dataset, pgexp_mlp)
            it_took = perf_counter() - start_time
            #print("  correct predicted_labels: ", predicted_labels)
            edge_masks_binarized = edge_masks[0] > self.importance_threshold

            index_list = []
            adding = []
            banned = []
            for i, (start, target) in enumerate(zip(your_dataset[0].edge_index[0], your_dataset[0].edge_index[1])):
                if [start, target] not in banned:
                    adding.append([start, target])
                    index_list.append(i)
                    banned.append([target, start])
            #print("len(adding): ", len(adding), "len(banned): ", len(banned), "len(index_list):", len(index_list))
            #print(edge_masks_binarized[index_list])

            saliency_maps = {}
            importance_dict = {}
            edge_masks = edge_masks[0].tolist()
            for i in range(len(index_list)):
                saliency_maps[i] = edge_masks[index_list[i]]

            for i in range(len(index_list)):
                importance_dict[i] = edge_masks_binarized.tolist()[index_list[i]]

            return it_took, saliency_maps, importance_dict
        else:
            print("recheck")
            return None, None







#dataset = TUDataset(root='data/TUDataset', name='MUTAG')
#target_class = 'correct'
#ExTrain_or_ExTest = 'test'
#importance_threshold = 0.3

#EXP = PGExplainer(Model_Name="GCN_plus_GAP", Explainability_name='PGExplainer', Task_name='Graph Classification',
#                  classifier_load_index=200, explainer_save_index=10000, Exp_Epoch=100, classifier_save_index=200,
#                  Exp_lr=0.001, input_dim=len(dataset[0].x[0]), hid_dim=7, output_dim=2, importance_threshold=importance_threshold,
#                  ExTrain_or_ExTest=ExTrain_or_ExTest, Exp_Load_index=100, your_dataset=[dataset[0]], target_class=target_class,
#                  DataSet_name="MUTAG")
#edge_mask = EXP(ExTrain_or_ExTest=ExTrain_or_ExTest, Exp_Load_index=100, your_dataset=[dataset[0]], target_class=target_class)
#print("edge_mask: ", edge_mask)
#print("saliency_maps: ", EXP.saliency_maps)
#print("EXP.it_took: ", EXP.it_took)
#print("importance_dict: ", EXP.importance_dict)


# print("Before Training Explainer: ", self.GNN_Model.GConvs[0].lin.weight)
# print("After Training Explainer: ", self.GNN_Model.GConvs[0].lin.weight)
# t1_start = perf_counter()
# EXP(ExTrain_or_ExTest=ExTrain_or_ExTest, Exp_Load_index=100, your_dataset=test_dataloader, target_class="correct")
# EXP(ExTrain_or_ExTest=ExTrain_or_ExTest, Exp_Load_index=100, your_dataset=test_dataloader, target_class="incorrect")
# t2_start = perf_counter()
# print("duration: ", (t2_start - t1_start)/len(test_dataset))


#adding = []
#banned = []
#for i, j in zip(edge_index[0], edge_index[1]):
#    if [i, j] not in banned:
#        adding.append([i, j])
#        Final_Embeddings.append(torch.cat([embedding[i], embedding[j]], dim=-1))
#        banned.append([j, i])