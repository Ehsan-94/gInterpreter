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
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn.parameter import Parameter
import numpy as np
from time import perf_counter
from torch.nn import Linear, LayerNorm
from torch import sigmoid
from scipy.special import softmax
from statistics import mean
import math


class GraphMask(object):
    def __init__(self, Model_Name, Explainability_name, Task_name, classifier_load_index, explainer_save_index,
                 Exp_Epoch, Exp_lr, explainer_hid_dim, input_dim, hid_dim, output_dim, DataSet_name,
                 importance_threshold, Exp_Load_index, ExTrain_or_ExTest, your_dataset, target_class):

        ######################              MAIN
        self.beta = 1 / 3
        self.gamma = -0.2
        self.zeta = 1.0
        self.fix_temp = True
        self.loc_bias = 3
        self.penalty_scaling = 1
        self.allowance = 0.03
        self.max_allowed_performance_diff = 0.05
        self.temp = self.beta if self.fix_temp else Parameter(torch.zeros(1).fill_(self.beta))
        self.gamma_zeta_ratio = np.math.log(-self.gamma / self.zeta)
        ###########
        self.Explainability_name = Explainability_name
        self.Task_name = Task_name
        self.Model_Name = Model_Name
        self.ExTrain_or_ExTest = ExTrain_or_ExTest
        self.importance_threshold = importance_threshold
        self.classifier_load_index = classifier_load_index
        self.DataSet_name = DataSet_name
        self.GNN_Model = self.load_model(Task_name=Task_name, Explainability_name=Explainability_name,
                                         Model_Name=Model_Name, classifier_load_index=classifier_load_index,
                                         input_dim=input_dim, hid_dim=hid_dim, output_dim=output_dim)
        self.input_dim = input_dim
        self.explainer_lr = Exp_lr
        self.explainer_hid_dim = explainer_hid_dim
        self.graphmask_mlp = Sequential(Linear(self.input_dim * 2, self.explainer_hid_dim),
                                        LayerNorm(self.explainer_hid_dim), ReLU(), Linear(self.explainer_hid_dim, 1))
        self.graphmask_mlp_optimizer = torch.optim.Adam(self.graphmask_mlp.parameters(), lr=self.explainer_lr)
        self.explainer_epochs = Exp_Epoch
        # self.criterion = torch.nn.CrossEntropyLoss()
        # self.criterion = F.binary_cross_entropy_with_logits
        self.criterion = nn.L1Loss()
        self.explainer_save_index = explainer_save_index
        self.Exp_Load_index = Exp_Load_index
        self.your_dataset = your_dataset


        self.baseline = torch.FloatTensor(1)
        bl_stdv = 1. / math.sqrt(1)
        self.baseline.uniform_(-bl_stdv, bl_stdv)
        self.baseline = torch.nn.Parameter(self.baseline, requires_grad=True)

        ###############.      Lagrangian Optimization
        self.min_alpha = -2
        self.max_alpha = 30
        self.update_counter = 0
        self.init_alpha = 0.55
        self.alpha_optimizer_lr = 1e-2
        self.alpha = torch.tensor(self.init_alpha, requires_grad=True)
        self.optimizer_alpha = torch.optim.RMSprop([self.alpha], lr=self.alpha_optimizer_lr, centered=True)
        self.update_counter = 0
        self.it_took, self.saliency_maps, self.importance_dict = self.drop_important_nodes(self.ExTrain_or_ExTest,
                                                                                           self.Exp_Load_index,
                                                                                           self.your_dataset,
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
                                                   hidden_dim=hid_dim, output_dim=output_dim,
                                                   num_hid_layers=2, Bias=True, act_fun='eLu', Weight_Initializer=1,
                                                   dropout_rate=0.1)
            return GNN_Model

    def loading_config(self, Task_name, Explainability_name, Model_Name, classifier_load_index, input_dim, hid_dim,
                       output_dim):
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

    def binary_concrete(self, explaier_outputs, temperature, summarize_penalty=True):
        explaier_outputs = explaier_outputs + self.loc_bias
        if self.ExTrain_or_ExTest == 'train':
            u = torch.empty_like(explaier_outputs).uniform_(1e-6, 1.0 - 1e-6)
            s = sigmoid((torch.log(u) - torch.log(1 - u) + explaier_outputs) / temperature)
            penalty = sigmoid(explaier_outputs - temperature * self.gamma_zeta_ratio)
        else:
            s = sigmoid(explaier_outputs)
            penalty = torch.zeros_like(explaier_outputs)

        if summarize_penalty:
            penalty = penalty.mean()

        s = s * (self.zeta - self.gamma) + self.gamma
        clipped_s = s.clamp(min=0, max=1)

        hard_concrete = (clipped_s > 0.5).float()
        clipped_s = clipped_s + (hard_concrete - clipped_s).detach()
        clipped_s = clipped_s.squeeze(dim=-1)

        return clipped_s, penalty

    def lagrangian_optimization_update(self, f, g, batch_size_multiplier):

        if batch_size_multiplier is not None and batch_size_multiplier > 1:
            if self.update_counter % batch_size_multiplier == 0:
                self.graphmask_mlp_optimizer.zero_grad()
                self.optimizer_alpha.zero_grad()

            self.update_counter += 1
        else:
            self.graphmask_mlp_optimizer.zero_grad()
            self.optimizer_alpha.zero_grad()

        loss = f + torch.nn.functional.softplus(self.alpha) * g
        loss.backward(retain_graph=True)

        if batch_size_multiplier is not None and batch_size_multiplier > 1:
            if self.update_counter % batch_size_multiplier == 0:
                self.graphmask_mlp_optimizer.step()
                self.alpha.grad *= -1
                self.optimizer_alpha.step()
        else:
            self.graphmask_mlp_optimizer.step()
            self.alpha.grad *= -1
            self.optimizer_alpha.step()

        if self.alpha.item() < -2:
            self.alpha.data = torch.full_like(self.alpha.data, -2)
        elif self.alpha.item() > 30:
            self.alpha.data = torch.full_like(self.alpha.data, 30)

    def train_step_explainer(self, merged_embeddings_list_batchs, GNN_Model, your_dataset, GNN_Model_preds_NOT_MASKED,
                             target_class):

        # lagrangian_optimization = LagrangianOptimization(self.graphmask_mlp_optimizer, batch_size_multiplier=your_dataset.batch_size)
        self.graphmask_mlp.train()
        self.graphmask_mlp.zero_grad()
        for batched_merged_embeddings, batched_preds_NOT_MASKED, batched_graphs in zip(merged_embeddings_list_batchs,
                                                                                       GNN_Model_preds_NOT_MASKED,
                                                                                       your_dataset):
            explaier_outputs = self.graphmask_mlp(batched_merged_embeddings).view(-1)

            edge_mask, sparsity_penalty = self.binary_concrete(explaier_outputs, self.temp)
            # print("edge_mask: ", edge_mask)
            # print("penalty: ", sparsity_penalty)
            # print("baseline: ", baseline)
            importance_indices = edge_mask == 0
            edge_mask[importance_indices] = self.baseline
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

            g = torch.relu(batch_loss - self.allowance).mean()
            f = (sparsity_penalty * self.penalty_scaling)

            self.lagrangian_optimization_update(f=f, g=g, batch_size_multiplier=your_dataset.batch_size)

            batch_loss.requires_grad = True
            batch_loss.backward(retain_graph=True)
            self.graphmask_mlp_optimizer.step()

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
            edge_mask = self.train_step_explainer(merged_embeddings_list, GNN_Model, your_dataset,
                                                  GNN_Model_preds_NOT_MASKED, target_class)
            edge_masks_per_epoch.append(edge_mask)

            if (epoch + 1) == self.explainer_save_index:
                torch.save({'epoch': epoch + 1, 'model_state_dict': self.graphmask_mlp.state_dict(),
                            'optimizer_state_dict': self.graphmask_mlp_optimizer.state_dict(),
                            'baseline_state_dict': self.baseline},
                           "/content/drive/My Drive/Explainability Methods/" + str(
                               self.Explainability_name) + " on " + str(self.Task_name) + "/Model/" + str(
                               self.Model_Name) + "_Model_classifier_GraphMask_MLP_" + str(
                               epoch + 1) + "_epochs_" + str(target_class) + ".pt")
                # torch.save({'epoch': epoch+1, 'baseline_state_dict': self.baseline}, "/content/drive/My Drive/Explainability Methods/" + str(self.Explainability_name) + " on " + str(self.Task_name) + "/Model/" + str(self.Model_Name) + "_BaseLine_Model_classifier_GraphMask_MLP_" + str(epoch + 1) + "_epochs_" + str(target_class) + ".py")
        self.clear_masks(GNN_Model)

    def test_explainer(self, GNN_Model, your_dataset, graphmask_mlp):
        predicted_labels_MASKED = []
        merged_embeddings_list_batchs = self.get_merged_embeddings(GNN_Model, your_dataset)
        GNN_Model_preds_NOT_MASKED = []
        edge_masks = []
        graphmask_mlp.eval()
        for batch_of_graphs in your_dataset:
            Output_of_Hidden_Layers_NOT_MASKED, pooling_layer_output_NOT_MASKED, ffn_output_NOT_MASKED, soft_NOT_MASKED = GNN_Model(
                batch_of_graphs)
            GNN_Model_preds_NOT_MASKED.append(soft_NOT_MASKED)

        for batched_merged_embeddings, batched_preds_NOT_MASKED, batched_graphs in zip(merged_embeddings_list_batchs,
                                                                                       GNN_Model_preds_NOT_MASKED,
                                                                                       your_dataset):
            explaier_outputs = graphmask_mlp(batched_merged_embeddings).view(-1)

            edge_mask, sparsity_penalty = self.binary_concrete(explaier_outputs, self.temp)
            edge_masks.append(edge_mask)

            importance_indices = edge_mask == 0
            edge_mask[importance_indices] = self.baseline

            self.apply_masks(GNN_Model, edge_mask, batched_graphs.edge_index, apply_sigmoid=True)

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
        Zs = [embedding[edge_index[0]], embedding[edge_index[1]]]
        return torch.cat(Zs, dim=-1)

    def get_hopped_embeddings(self, GNN_Model, batch_of_graphs):
        GNN_Model.eval()
        Output_of_Hidden_Layers, pooling_layer_output, ffn_output, soft = GNN_Model(batch_of_graphs)
        GNN_Model_test_pred = soft.argmax(dim=1)
        return Output_of_Hidden_Layers[-1], GNN_Model_test_pred

    def load_explainer_mlp(self, Exp_Load_index, target_class):
        graphmask_mlp = Sequential(Linear(self.input_dim * 2, self.explainer_hid_dim),
                                   LayerNorm(self.explainer_hid_dim), ReLU(), Linear(self.explainer_hid_dim, 1))
        graphmask_mlp_optimizer = torch.optim.Adam(self.graphmask_mlp.parameters(), lr=self.explainer_lr)
        checkpoint = torch.load(str(self.Model_Name) + "_Model_classifier_GraphMask_MLP_" + str(Exp_Load_index) + "_epochs_" + str(target_class) + ".pt")
        graphmask_mlp.load_state_dict(checkpoint['model_state_dict'])
        graphmask_mlp_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        baseline = checkpoint['baseline_state_dict']
        print("BaseLine: ", target_class, "   ", baseline)

        # py_path2 = "/content/drive/My Drive/Explainability Methods/GraphMask on Graph Classification/Model/"
        # sys.path.insert(0,py_path2)

        # import GCN_plus_GAP_BaseLine_Model_classifier_GraphMask_MLP_100_epochs_correct as basline_file
        # x = basline_file['baseline_state_dict']
        # baseline = torch.FloatTensor(1)
        # bl_stdv = 1. / math.sqrt(1)
        # baseline.uniform_(-bl_stdv, bl_stdv)
        # baseline = torch.nn.Parameter(baseline, requires_grad=True)
        # baseline.load()

        return graphmask_mlp, graphmask_mlp_optimizer, baseline

    def drop_important_nodes(self, ExTrain_or_ExTest, Exp_Load_index, your_dataset, target_class):
        self.ExTrain_or_ExTest = ExTrain_or_ExTest
        if ExTrain_or_ExTest == "train":
            t1 = perf_counter()
            self.train_explainer(self.GNN_Model, your_dataset, target_class)
            t2 = perf_counter()
            print((t2 - t1) / len(your_dataset))
        elif ExTrain_or_ExTest == "test":
            graphmask_mlp, graphmask_mlp_optimizer, baseline = self.load_explainer_mlp(Exp_Load_index=Exp_Load_index,
                                                                                       target_class="correct")
            start_time = perf_counter()
            predicted_labels, edge_masks = self.test_explainer(self.GNN_Model, your_dataset, graphmask_mlp)
            it_took = perf_counter() - start_time
            print("edge_masks: ", edge_masks)

            edge_masks_binarized = edge_masks[0] > self.importance_threshold

            index_list = []
            adding = []
            banned = []
            for i, (start, target) in enumerate(zip(your_dataset[0].edge_index[0], your_dataset[0].edge_index[1])):
                if [start, target] not in banned:
                    adding.append([start, target])
                    index_list.append(i)
                    banned.append([target, start])

            saliency_maps = {}
            importance_dict = {}
            edge_masks = edge_masks[0].tolist()
            for i in range(len(index_list)):
                saliency_maps[i] = edge_masks[index_list[i]]

            for i in range(len(index_list)):
                importance_dict[i] = edge_masks_binarized.tolist()[index_list[i]]

            #print("importance_dict: ",importance_dict)
            #print("saliency_maps: ", saliency_maps)
            return it_took, saliency_maps, importance_dict
            #print("  correct predicted_labels: ", predicted_labels)

            #graphmask_mlp, graphmask_mlp_optimizer, baseline = self.load_explainer_mlp(Exp_Load_index=Exp_Load_index,
            #                                                                           target_class="incorrect")
            #predicted_labels = self.test_explainer(self.GNN_Model, your_dataset, graphmask_mlp)
            #print("incorrect predicted_labels: ", predicted_labels)
        else:
            print("recheck")
            return None, None, None


#target_class = 'correct'
#ExTrain_or_ExTest = 'test'
#DataSet_name = "MUTAG"
#importance_threshold = 0.3

#if ExTrain_or_ExTest == 'test':
#    dataset = TUDataset(root='data/TUDataset', name='MUTAG')


#t1_start = perf_counter()
#EXP = GraphMask(Model_Name="GCN_plus_GAP", Explainability_name='GraphMask', Task_name='Graph Classification',
#                classifier_load_index=200, explainer_save_index=50, Exp_Epoch=50, Exp_lr=0.001,
#                explainer_hid_dim=7, input_dim=len(dataset[0].x[0]), hid_dim=7, output_dim=2,
#                DataSet_name=DataSet_name, importance_threshold=importance_threshold, Exp_Load_index=100,
#                ExTrain_or_ExTest=ExTrain_or_ExTest, your_dataset=[dataset[0]], target_class=target_class)
#EXP(ExTrain_or_ExTest=ExTrain_or_ExTest, Exp_Load_index=100, your_dataset=[dataset[0]], target_class=target_class)
#t2_start = perf_counter()
#attribution_time = (t2_start - t1_start) / len(dataset)
#print("attribution_time: ", attribution_time)

