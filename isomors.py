from networkx.algorithms import isomorphism
import networkx as nx
from torch_geometric.datasets import TUDataset
dataset = TUDataset(root='data/TUDataset', name='MUTAG')
g1 = dataset[0]
g2 = dataset[1]

G = nx.Graph(
    [
        ("a", "g"),
        ("a", "h"),
        ("a", "i"),
        ("g", "b"),
        ("g", "c"),
        ("b", "h"),
        ("b", "j"),
        ("h", "d"),
        ("c", "i"),
        ("c", "j"),
        ("i", "d"),
        ("d", "j"),
    ]
)
#print(G.nodes)
########################################################################################################################
def graphize(Graph):
    my_nodes_name_index = []
    for nods in Graph.x:
        my_nodes_name_index.append(nods.detach().tolist().index(max(nods)))

    my_edges = []
    for nods in Graph.edge_attr:
        my_edges.append(nods.detach().tolist().index(max(nods)))

    my_nodes_dict = {}
    for i in range(len(my_nodes_name_index)):
        my_nodes_dict[str(i)] = dict(color=str(my_nodes_name_index[i]), pos='')

    edges = []
    for i in range(len(Graph.edge_index[0])):
        edges.append((str(int(Graph.edge_index[0][i].detach())), str(int(Graph.edge_index[1][i].detach())),
                      my_edges[i]))
    original_graph = nx.Graph()
    original_graph.add_nodes_from(
        [(n, {"type": atom_index}) for n, atom_index in zip(my_nodes_dict, my_nodes_name_index)])
    original_graph.add_edges_from((u, v, {"type": label}) for u, v, label in edges)


    return original_graph


def find_isomers(graph_index, your_dataset):
    target_graph = graphize(your_dataset[graph_index])
    print("target graph: ", target_graph.nodes)


    print(your_dataset[graph_index])
    isomers_of_the_target_graph = []
    for i, graph in enumerate(your_dataset):
        new_graph = graphize(graph)
        if nx.vf2pp_is_isomorphic(target_graph, new_graph, node_label=None) and i != graph_index:
            isomers_of_the_target_graph.append(1)
        else:
            isomers_of_the_target_graph.append(0)
    return isomers_of_the_target_graph

print(find_isomers(100, dataset))

number_of_isomers_for_each_graph = []
for i, graph in enumerate(dataset):
    number_of_isomers_for_each_graph.append(sum(find_isomers(i, dataset)))
    break

print(number_of_isomers_for_each_graph)
print(type(number_of_isomers_for_each_graph))




#new_g1 = grphize(g1)
#new_g2 = grphize(g2)

#res = nx.vf2pp_is_isomorphic(0, dataset, node_label=None)
#print(res)