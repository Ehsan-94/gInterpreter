import torch
from torch_geometric.nn import MessagePassing
from torch.nn.parameter import Parameter
import math
import GCN_plus_GAP as Graph_Network
from torch_geometric.datasets import TUDataset


class CF2_Explainer(torch.nn.Module):
    def __init__(self, graph, GNN_Model, reset_mask_to_None):
        super(CF2_Explainer, self).__init__()

        self.graph = graph
        self.num_nodes = len(self.graph.x)
        self.num_edges = len(self.graph.edge_index[0])
        self.GNN_Model = GNN_Model
        if reset_mask_to_None == True:
            self.clear_masks(self.GNN_Model)

        self.adj_mask = self.adj_mask_creation()

    def forward(self):
        masked_adj = self.get_masked_adj()

        self.apply_masks(self.GNN_Model, masked_adj)
        Output_of_Hidden_Layers, pooling_layer_output, ffn_output, pred_factual = self.GNN_Model(self.graph)  # factual

        self.apply_masks(self.GNN_Model, torch.ones_like(masked_adj) - masked_adj)
        Output_of_Hidden_Layers, pooling_layer_output, ffn_output, pred_c_factual = self.GNN_Model(
            self.graph)  # counterfactual

        return pred_factual, pred_c_factual

    def apply_masks(self, model, mask):
        loop_mask = self.graph.edge_index[0] != self.graph.edge_index[1]
        for i, module in enumerate(model.modules()):
            if isinstance(module, MessagePassing):
                mask = Parameter(mask)
                module._explain = True
                module._edge_mask = mask
                module._loop_mask = loop_mask
                module._apply_sigmoid = False
                # print(i, " module._edge_mask: ", module._edge_mask)

    def clear_masks(self, model):
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.explain = False
                module._edge_mask = None
                module._loop_mask = True
                module._apply_sigmoid = False
        return module

    def loss(self, pred_factual, pred_c_factual, original_pred, gamma_coeff, lambda_coeff, alpha_coeff):

        relu = torch.nn.ReLU()
        strength_factual = relu(original_pred + gamma_coeff - pred_factual)  # factual
        strength_c_factual = relu(original_pred + pred_c_factual - gamma_coeff)  # counterfactual

        masked_adj = self.get_masked_adj()

        L1 = torch.linalg.norm(masked_adj[0])
        strength_loss = lambda_coeff * (alpha_coeff * strength_factual + (1 - alpha_coeff) * strength_c_factual)
        loss = L1 + strength_loss
        # print("L1: ", L1, "    ", " Strength_Loss: ", strength_loss)
        return strength_factual, strength_c_factual, L1, loss

    def adj_mask_creation(self):
        mask_for_adj = torch.nn.Parameter(torch.FloatTensor(self.num_edges))
        std = torch.nn.init.calculate_gain("relu") * math.sqrt(2.0 / (self.num_nodes + self.num_nodes))
        with torch.no_grad():
            mask_for_adj.normal_(1.0, std)
        return mask_for_adj

    def get_masked_adj(self):

        sym_mask = torch.sigmoid(self.adj_mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2

        mask = torch.reshape(sym_mask, (-1,))
        # print(mask)

        return mask

#dataset = TUDataset(root='data/TUDataset', name='MUTAG')
#GNN_Model2 = Graph_Network.GCN_plus_GAP(model_name='GCN_plus_GAP', model_level='graph', input_dim=7, hidden_dim=7,
#                                        output_dim=2, num_hid_layers=2, Bias=True, act_fun='eLu', Weight_Initializer=1,
#                                        dropout_rate=0.1)
#cf2_explainer_example = CF2_Explainer(graph=dataset[0], GNN_Model=GNN_Model2, reset_mask_to_None=True)
#pred_factual, pred_counter_factual = cf2_explainer_example.forward()
#print("pred_factual: ", pred_factual)
#
#print("pred_counter_factual: ", pred_counter_factual)




