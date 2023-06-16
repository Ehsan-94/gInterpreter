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
import statistics
from scipy import stats
import logging
import pandas as pd



class PGM_Graph_Explainer(object):
    def __init__(self, Model_Name, classifier_load_index, input_dim, hid_dim, output_dim, graph, perturb_feature_list,
                 perturb_mode, perturb_indicator, DataSet_name, importance_threshold):
        self.Model_Name = Model_Name
        self.Task_name = 'Graph Classification'
        self.Explainability_name = "PGMExplainer"
        self.DataSet_name = DataSet_name
        self.your_model = self.load_model(Task_name=self.Task_name, Explainability_name=self.Explainability_name,
                                          Model_Name=Model_Name,
                                          classifier_load_index=classifier_load_index, input_dim=input_dim,
                                          hid_dim=hid_dim, output_dim=output_dim)
        self.your_model.eval()

        self.graph = graph
        self.num_layers = 2
        self.perturb_feature_list = perturb_feature_list
        self.perturb_mode = perturb_mode
        self.perturb_indicator = perturb_indicator
        self.node_feat = graph.x.numpy()
        self.top_counted_nodes = int(len(self.graph.x) - importance_threshold * len(self.graph.x))
        self.pgm_nodes, self.p_values, self.candidate_nodes, self.dependent_nodes, self.importance_dict, \
            self.saliency_maps, self.it_took = self.explain(num_samples=len(self.graph.x), percentage=50,
                                                                     top_node=self.top_counted_nodes, p_value_threshold=0.05,
                                                                     pred_threshold=0.1, ctg='correct')
        #print("self.importance_dict: ", self.importance_dict)
        #print("self.saliency_maps: ", self.saliency_maps)
        #print("self.it_took : ", self.it_took)
        #print("pgm_nodes: ", self.pgm_nodes, " p_values: ", self.p_values, " candidate_nodes: ", self.candidate_nodes,
        #      "dependent_nodes: ", self.dependent_nodes)

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
        GNN_Model = Graph_Network.GCN_plus_GAP(model_name=Model_Name, model_level='graph', input_dim=input_dim,
                                               hidden_dim=hid_dim, output_dim=output_dim,
                                               num_hid_layers=2, Bias=True, act_fun='eLu', Weight_Initializer=1,
                                               dropout_rate=0.1)
        optimizer = torch.optim.Adam(params=GNN_Model.parameters(), lr=0.001, weight_decay=1e-6)
        checkpoint = torch.load(Model_Name + " " + Explainability_name + " " + Task_name + " " + self.DataSet_name + " " + str(classifier_load_index) + ".pt")
        GNN_Model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return GNN_Model, optimizer, epoch

    def cressie_read(self, X, Y, Z, data_pertubed_Samples, significance_level):
        return self.power_divergence(X=X, Y=Y, Z=Z, data_pertubed_Samples=data_pertubed_Samples, lambda_="cressie-read",
                                     significance_level=significance_level)

    def power_divergence(self, X, Y, Z, data_pertubed_Samples, lambda_, significance_level):
        if hasattr(Z, "__iter__"):
            Z = list(Z)
        else:
            raise (f"Z must be an iterable. Got object type: {type(Z)}")

        if (X in Z) or (Y in Z):
            raise ValueError(f"The variables X or Y can't be in Z. Found {X if X in Z else Y} in Z.")
        if len(Z) == 0:
            chi, p_value, dof, expected = stats.chi2_contingency(
                data_pertubed_Samples.groupby([X, Y]).size().unstack(Y, fill_value=0), lambda_=lambda_
            )
        else:
            chi = 0
            dof = 0
            for z_state, df in data_pertubed_Samples.groupby(Z):
                try:
                    c, _, d, _ = stats.chi2_contingency(df.groupby([X, Y]).size().unstack(Y, fill_value=0),
                                                        lambda_=lambda_)
                    chi += c
                    dof += d
                except ValueError:
                    if isinstance(z_state, str):
                        logging.info(f"Skipping the test {X} \u27C2 {Y} | {Z[0]}={z_state}. Not enough samples")
                    else:
                        z_str = ", ".join([f"{var}={state}" for var, state in zip(Z, z_state)])
                        logging.info(f"Skipping the test {X} \u27C2 {Y} | {z_str}. Not enough samples")
            p_value = 1 - stats.chi2.cdf(chi, df=dof)
        return chi, p_value, dof

    def perturb_node_features(self, node_feature_matrix, targeted_node_idx, random_perturbation_permission):

        graph_node_features = deepcopy(node_feature_matrix)
        targeted_node_feat_to_perturb_array = deepcopy(graph_node_features[targeted_node_idx])
        # print("targeted_node_feat_to_perturb_array: ", targeted_node_feat_to_perturb_array)
        epsilon = 0.05 * np.max(self.node_feat, axis=0)

        if random_perturbation_permission == 1:
            for i in range(targeted_node_feat_to_perturb_array.shape[0]):
                if i in self.perturb_feature_list:
                    if self.perturb_mode == "mean":
                        targeted_node_feat_to_perturb_array[i] = np.mean(node_feature_matrix[:, i])
                    elif self.perturb_mode == "zero":
                        targeted_node_feat_to_perturb_array[i] = 0
                    elif self.perturb_mode == "max":
                        targeted_node_feat_to_perturb_array[i] = np.max(node_feature_matrix[:, i])
                    elif self.perturb_mode == "uniform":
                        targeted_node_feat_to_perturb_array[i] = targeted_node_feat_to_perturb_array[
                                                                     i] + np.random.uniform(low=-epsilon[i],
                                                                                            high=epsilon[i])
                        if targeted_node_feat_to_perturb_array[i] < 0:
                            targeted_node_feat_to_perturb_array[i] = 0
                        elif targeted_node_feat_to_perturb_array[i] > np.max(self.node_feat, axis=0)[i]:
                            targeted_node_feat_to_perturb_array[i] = np.max(self.node_feat, axis=0)[i]

        graph_node_features[targeted_node_idx] = targeted_node_feat_to_perturb_array

        return graph_node_features

    def gather_perturbed_node_features(self, sampling_count, index_to_perturb, percentage, p_value_threshold,
                                       pred_threshold):
        Output_of_Hidden_Layers, pooling_layer_output, ffn_output, pred_torch = self.your_model(self.graph)

        pred_label = pred_torch.argmax(dim=1)

        num_nodes_in_graph = self.node_feat.shape[0]
        # print("self.graph.x: ", self.graph.x)

        Samples = []
        for iteration in range(sampling_count):
            graph_original_features = deepcopy(self.node_feat)
            sample = []
            for node_index in range(num_nodes_in_graph):
                if node_index in index_to_perturb:
                    seed = np.random.randint(100)
                    if seed < percentage:
                        random_perturbation_permission = 1
                        graph_perturbed_features = self.perturb_node_features(
                            node_feature_matrix=graph_original_features, targeted_node_idx=node_index,
                            random_perturbation_permission=random_perturbation_permission)
                        # print("graph_perturbed_features: ", graph_perturbed_features)
                    else:
                        random_perturbation_permission = 0
                else:
                    random_perturbation_permission = 0
                sample.append(random_perturbation_permission)

                perturbed_graph = deepcopy(self.graph)
                if random_perturbation_permission:
                    graph_perturbed_features_torch = torch.tensor(graph_perturbed_features, dtype=torch.float)
                    perturbed_graph.x = graph_perturbed_features_torch
                    # print("graph_perturbed_features_torch: ", graph_perturbed_features_torch)
                Output_of_Hidden_Layers, pooling_layer_output, ffn_output, pred_perturb_torch = self.your_model(
                    perturbed_graph)

                pred_change = max(pred_torch[0].tolist()) - pred_perturb_torch[0].tolist()[pred_label]

                sample.append(pred_change)
            Samples.append(sample)

        Samples = np.asarray(Samples)
        if self.perturb_indicator == "abs":
            Samples = np.abs(Samples)
        # print("Samples: ", np.array(Samples).shape)
        top = int(sampling_count / 8)
        top_idx = np.argsort(Samples[:, num_nodes_in_graph])[-top:]
        # print("top_idx: ", top_idx)
        # print("Samples[:, num_nodes_in_graph]: ", Samples[:, num_nodes_in_graph])
        for i in range(sampling_count):
            if i in top_idx:
                Samples[i, num_nodes_in_graph] = 1
            else:
                Samples[i, num_nodes_in_graph] = 0

        return Samples

    def explain(self, num_samples, percentage, top_node, p_value_threshold, pred_threshold, ctg):

        t1 = perf_counter()
        if top_node == None:
            top_node = int(self.node_feat.shape[0] / 8)

        #         Round 1
        Samples = self.gather_perturbed_node_features(sampling_count=num_samples,
                                                      index_to_perturb=range(self.node_feat.shape[0]),
                                                      percentage=percentage,
                                                      p_value_threshold=p_value_threshold,
                                                      pred_threshold=pred_threshold)
        # print(len(Samples[0]), " Samples: ", list(Samples))
        data_pertubed_Samples1 = pd.DataFrame(Samples)
        # est = ConstraintBasedEstimator(data)

        p_values = []
        candidate_nodes = []
        # The entry for the graph classification data is at "num_nodes"
        for node in range(self.node_feat.shape[0]):
            chi2, p, dof = self.cressie_read(X=node, Y=self.node_feat.shape[0], Z=[],
                                             data_pertubed_Samples=data_pertubed_Samples1, significance_level=0.05)
            # print("this is returned P: ", p)
            p_values.append(p)

        number_candidates = top_node
        candidate_nodes = np.argpartition(p_values, number_candidates)[0:number_candidates]

        #         Round 2
        Samples = self.gather_perturbed_node_features(sampling_count=num_samples, index_to_perturb=candidate_nodes,
                                                      percentage=percentage,
                                                      p_value_threshold=p_value_threshold,
                                                      pred_threshold=pred_threshold)
        data = pd.DataFrame(Samples)
        # est = ConstraintBasedEstimator(data)

        p_values = []
        dependent_nodes = []

        for node in range(self.node_feat.shape[0]):
            chi2, p, dof = self.cressie_read(X=node, Y=self.node_feat.shape[0], Z=[], data_pertubed_Samples=data,
                                             significance_level=0.05)
            # chi2, p = chi_square(node, target, [], data)
            p_values.append(p)
            if p < p_value_threshold:
                dependent_nodes.append(node)
        t2 = perf_counter()

        if ctg == 'correct':
            top_p = np.min((top_node, self.node_feat.shape[0] - 1))
        elif ctg == 'incorrect':
            top_p = np.max((top_node, self.node_feat.shape[0] - 1))
        ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
        pgm_nodes = list(ind_top_p)
        saliency_maps = {}
        importance_dict = {}
        importance_dict[0] = {}
        for i in range(len(self.graph.x)):
            saliency_maps[i] = p_values[i]
            if i in candidate_nodes:
                importance_dict[0][i] = True
            else:
                importance_dict[0][i] = False

        return pgm_nodes, p_values, candidate_nodes, dependent_nodes, importance_dict, saliency_maps, t2-t1

#dataset = TUDataset(root='data/TUDataset', name='MUTAG')
#input_graph = dataset[0]
#DataSet_name = "MUTAG"
#pgmx = PGM_Graph_Explainer(Model_Name="GCN_plus_GAP", classifier_load_index=200, input_dim=7, hid_dim=7, output_dim=2,
#                           graph=input_graph, DataSet_name=DataSet_name,
#                           perturb_feature_list=[None], perturb_mode="mean", perturb_indicator="abs", importance_threshold=0.5)

#pgm_nodes, p_values, candidate_nodes, dependent_nodes, importance_dict, saliency_maps, time = pgmx.explain(num_samples=len(input_graph.x), percentage=50,
#                                                                     top_node=0, p_value_threshold=0.05,
#                                                                     pred_threshold=0.1, ctg='correct')
#print("pgm_nodes: ", pgm_nodes, " p_values: ", p_values, " candidate_nodes: ", candidate_nodes, "dependent_nodes: ",
#      dependent_nodes)