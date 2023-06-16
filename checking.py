import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torch_geometric.datasets import TUDataset
from sklearn.manifold import TSNE
from random import shuffle
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import random
import numpy as np
#from copy import deepcopy
#Mutag_Dataset = TUDataset(root='data/TUDataset', name='MUTAG')
#print(Mutag_Dataset[0].x)
#index = 0
#Num_Output_Dimensions = 3
#Mutag_Reduced_D = TSNE(n_components=Num_Output_Dimensions).fit_transform(Mutag_Dataset[index].x.detach().cpu().numpy())
#print(Mutag_Reduced_D)


'''
def My_DataStructure(Graph):
    my_nodes = []
    my_nodes_color = []
    for nods in Graph.x:
        my_nodes.append(nods.detach().tolist().index(max(nods)))
        my_nodes_color.append(nods.detach().tolist().index(max(nods)))
    #print(my_nodes)
    my_edges = []
    for nods in Graph.edge_attr:
        my_edges.append(nods.detach().tolist().index(max(nods)))
    #print(my_edges)
    color_dict = {0: "Red", 1: "Blue", 2: "Yellow", 3: "Green", 4: "Gray", 5: "Purple", 6: "Pink"}
    my_nodes_dict = {}
    for i in range(len(my_nodes)):
        my_nodes_dict[str(i)] = dict(pos='')


    bond_dict = {0: "aromatic", 1: "single", 2: "double", 3: "triple"}
    # list to define edges
    edges = []
    for i in range(len(Graph.edge_index[0])):
       edges.append((str(int(Graph.edge_index[0][i].detach())), str(int(Graph.edge_index[1][i].detach())), bond_dict[my_edges[i]]))
    return Graphize(my_nodes_dict, my_nodes_color, edges)

#print("my nodes: ", len(my_nodes_dict), my_nodes_dict)
#print('my edges: ', len(edges), edges)
def Graphize(my_nodes_dict, my_nodes_color, edges):
    #print(my_nodes_dict)
    original_graph = nx.Graph()
    # add nodes and edges
    original_graph.add_nodes_from([n for n in my_nodes_dict])
    #print('nodes: ', len(original_graph.nodes), original_graph.nodes)
    original_graph.add_edges_from((u, v, {"type": label}) for u, v, label in edges)
    #print('edges: ', len(original_graph.edges), original_graph.edges)

    posit = nx.spring_layout(original_graph, dim=3)
    print('my posit: ', posit)

    return Update_Graph(original_graph, my_nodes_color, posit)

#for node in original_graph.nodes:

#print("My Need: ", posit)
#print(posit.items())
#print(len(posit.values()))
#print(original_graph.nodes[0]['pos'])
#for i in range(len(original_graph.nodes)):
    #print(posit[str(i)])
#    original_graph.nodes[str(i)]=dict(pos=posit[str(i)])



#print(original_graph)
#print("Nodes: ", original_graph.nodes)
#print("Edges: ", original_graph.edges)
#print(original_graph)
def Update_Graph(original_graph, my_nodes_color, posit):
    for i in range(len(original_graph.nodes)):
       original_graph.nodes[str(i)].update(pos=posit[str(i)])

    edge_trace = Edges_Interactive_Part(original_graph)
    node_trace = Nodes_Interactive_Part(original_graph)

    node_trace = Adjacency(original_graph, my_nodes_color, node_trace)
    fig = Figure_Drawing(edge_trace, node_trace)
    return fig

#print("mine ", original_graph.nodes['0'], original_graph.nodes['0']['pos'])


def Edges_Interactive_Part(original_graph):
    edge_x = []
    edge_y = []
    edge_z = []

    for edge in original_graph.edges():
        #print(edge)
        x0, y0, z0 = original_graph.nodes[edge[0]]['pos']
        x1, y1, z1 = original_graph.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=0.8, color='rgb(125,125,125)'),
        hoverinfo='none',
        mode='lines')
    return edge_trace


def Nodes_Interactive_Part(original_graph):
    node_x = []
    node_y = []
    node_z = []
    for node in original_graph.nodes():
        x, y, z = original_graph.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale=[[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']],
            reversescale=True,
            #color=["Red", "Blue", "Yellow", "Green", "Gray", "Purple", "Pink"],
            color=[],
            size=20,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    return node_trace

def Adjacency(original_graph, my_nodes_color, node_trace):
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(original_graph.adjacency()):
        #node_adjacencies.append(len(adjacencies[1]))
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))
        #print(original_graph.nodes[str(node)]['pos'])


    #print(node_adjacencies)
    #print(my_nodes_color)
    #node_trace.marker.color = node_adjacencies#[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    node_trace.marker.color = my_nodes_color
    node_trace.text = node_text

    return node_trace

def Figure_Drawing(edge_trace, node_trace):
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=True,
                title=''
                )
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Network graph made with Python',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis)),
                        annotations=[dict(
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0, y=0, xanchor='right', yanchor='top'
                        )
                        ]
                    ))
    #fig.show()
    return fig


#xaxis = dict(showgrid=False, zeroline=False, showticklabels=False),
#yaxis = dict(showgrid=False, zeroline=False, showticklabels=False)

Mutag_Dataset = TUDataset(root='data/TUDataset', name='MUTAG')
Graph = Mutag_Dataset[0]
fig = My_DataStructure(Graph)
fig.show()
'''
'''
MUTAG_dataset = TUDataset(root='data/TUDataset', name='MUTAG')
gr = MUTAG_dataset[0]
edge_df = MUTAG_dataset[0].edge_index
node_df = MUTAG_dataset[0].x


edge_df = edge_df.detach().cpu().numpy()
edge_df = edge_df.tolist()

import random
node_names = []
edge_weight = []
for i in range(len(gr.x)):
  node_names.append(i)

for i in range(len(edge_df[0])):
  edge_weight.append(random.randint(1,10))

print(edge_weight)
print(len(edge_weight))

edge_df_merged = [(edge_df[0][i], edge_df[1][i], edge_weight[i]) for i in range(0,len(edge_df[0]))]
import pandas as pd
edge_df = pd.DataFrame (edge_df_merged, columns = ['from', 'to', 'weight'])

from pyvis.network import Network
def my_method():


    node_names = []

    for i in range(len(gr.x)):
        node_names.append(i)



    net = Network(notebook=True)

    #print(xs)
    #["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen"]
    list_string = map(str, node_names)
    list_string = list(list_string)
    print(len(node_names))
    print(list(list_string))

    net.add_nodes(node_names, label=list_string, color=['#3da831', '#9a31a8', '#3155a8', '#eb4034', '#3da831', '#9a31a8', '#3155a8', '#eb4034', '#3da831', '#9a31a8', '#3155a8', '#eb4034', '#3da831', '#9a31a8', '#3155a8', '#eb4034', '#3da831'])
    net.add_edges(edge_df_merged)
    #net.add_edge(15, 16)

    net.repulsion(node_distance=100, spring_length=0)
    return net


n = my_method()
n.show()
'''
print('================================================================================================================')
'''
mut_edge = np.loadtxt('MUTAG_edge_labels.txt')
mut_node = np.loadtxt('MUTAG_node_labels.txt')
A= np.loadtxt('MUTAG_A.txt', dtype=str)
#print("mut nodes: ", max(mut_node))

color_dict= {0:"Red", 1:"Blue", 2:"Yellow",3:"Green", 4:"Gray", 5:"Purple", 6:"Pink"}
# create dict for nodes
node={}
for i in range(len(my_nodes)):
   node[str(i)]=dict(color=color_dict[mut_node[i]])



# dict for edge labels
bond_dict = {0:"aromatic", 1:"single", 2:"double", 3:"triple"}
# list to define edges
edges=[]
for i in range(7440):
   edges.append((A[i][0].split(',')[0], A[i][1], bond_dict[mut_edge[i]]))

original_graph = nx.Graph()
# add nodes and edges
original_graph.add_nodes_from(n for n in node.items())
original_graph.add_edges_from((u, v, {"type": label}) for u, v, label in edges)
####################     TEST
G = nx.random_geometric_graph(20, 0.125)
#G=original_graph
print(G)
print("Nodes: ", G.nodes)
print("Edges: ", G.edges)
print(G)
edge_x = []
edge_y = []
print("items ", G.nodes[0], G.nodes[0]['pos'])

for edge in G.edges():
    #print(edge)
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.8, color='red'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='Reds',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append('# of connections: '+str(len(adjacencies[1])))

node_trace.marker.color = node_adjacencies
node_trace.text = node_text

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.show()
'''
###########################
'''
Mutag_Dataset = TUDataset(root='data/TUDataset', name='MUTAG')
color_dict = {0: "Red", 1: "Blue", 2: "Yellow", 3: "Green", 4: "Gray", 5: "Purple", 6: "Pink"}
#print("Number of Edges: ", len(mut_edge), "Max: ", max(mut_edge), "Min: ", min(mut_edge))
#print("Number of Nodes: ", len(mut_node), "Max: ", max(mut_node), "Min: ", min(mut_node))

node = {}
for graph in Mutag_Dataset:
    for i in range(len(graph.x)):
        node[str(i)] = dict(color=color_dict[graph.x[i].detach().tolist().index(max(graph.x[i].detach().tolist()))])
print("Nodes: ", node)

bond_dict = {0: "aromatic", 1: "single", 2: "double", 3: "triple"}
edges = []
for graph in Mutag_Dataset:
    for i in range(len(graph.edge_index)):
        edges.append((graph.edge_index[0][i].detach().tolist(), graph.edge_index[1][i].detach().tolist(), bond_dict[graph.edge_attr[i].detach().tolist().index(max(graph.edge_attr[i].detach().tolist()))]))
print("Edges: ", edges)
original_graph = nx.Graph()
original_graph.add_nodes_from(n for n in node.items())
original_graph.add_edges_from((u, v, {"type": label}) for u, v, label in edges)
print("First Graph: ", original_graph.nodes)


'''



'''
Mutag_Dataset = TUDataset(root='data/TUDataset', name='MUTAG')




def sample_graphs_randomly(count):
    #Mutag_Dataset = shuffle(TUDataset(root='data/TUDataset', name='MUTAG'))
    Randomly_Taken_Graphs = []
    Index_of_Randomly_Taken_Graphs = []
    for i in range(count):
        r = random.randint(0, len(Mutag_Dataset) - 1)
        Index_of_Randomly_Taken_Graphs.append(r)
        Randomly_Taken_Graphs.append(Mutag_Dataset[r])
    return Randomly_Taken_Graphs, Index_of_Randomly_Taken_Graphs


def Reduce_Dimensionality(dataset, output_D):
    length = []
    #New_Dataset = deepcopy(dataset)
    New_Dataset = dataset
    for graph in dataset:
        length.append(len(graph.x))
    for i in range(len(dataset)):
        New_Dataset[i].x = (TSNE(n_components=output_D, init='pca', perplexity=min(length)-1).fit_transform(dataset[i].x.detach().cpu().numpy()).tolist())
    return New_Dataset


def Reduce_Dimensionality_of_Random_Graphs(Count_of_Random_Graphs, Num_Output_Dimensions):
    Randomly_Taken_Graphs, Index_of_Randomly_Taken_Graphs = sample_graphs_randomly(Count_of_Random_Graphs)
    Mutag_Reduced_D = Reduce_Dimensionality(Randomly_Taken_Graphs, Num_Output_Dimensions)
    return Mutag_Reduced_D

Count_of_Random_Graphs = 10
Num_Output_Dimensions = 3

Mutag_Reduced_D = Reduce_Dimensionality_of_Random_Graphs(Count_of_Random_Graphs, Num_Output_Dimensions)
#print(Mutag_Reduced_D[0].edge_index)

mut_edge = np.loadtxt('MUTAG_edge_labels.txt')
mut_node = np.loadtxt('MUTAG_node_labels.txt')
A= np.loadtxt('MUTAG_A.txt', dtype=str)
def merge_srource_target(dataset):
    Edges_Merged = []
    for graph in dataset:
        edges_each_graph = []
        for i in range(len(graph.edge_index[0])):
            edges_each_graph.append((graph.edge_index[0][i].detach().numpy().tolist(), graph.edge_index[1][i].detach().numpy().tolist()))
        Edges_Merged.append(edges_each_graph)
    return Edges_Merged
Edges = merge_srource_target(Mutag_Reduced_D)
#print(Mutag_Reduced_D)
print("Edges of the First graph: ", Edges[0])
'''

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import graphviz


iris = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
iris.head()

X = torch.tensor(iris.drop("variety", axis=1).values, dtype=torch.float)
y = torch.tensor(
    [0 if vty == "Setosa" else 1 if vty == "Versicolor" else 2 for vty in iris["variety"]],
    dtype=torch.long
)

print(X[:3])
print()
print(y[:3])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(in_features=4, out_features=16)
        self.hidden_1 = nn.Linear(in_features=16, out_features=16)
        self.output = nn.Linear(in_features=16, out_features=3)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_1(x))
        return self.output(x)


model = Net()
print(model)



from torchviz import make_dot

model = Net()
y = model(X)
'''

def Network_datastructure_part(fig_width, fig_height):
    number_of_layers = 2
    number_of_nodes_on_each_layer = [7, 7, 2]

    my_nodes_name_index_together = []
    for nods in range(sum(number_of_nodes_on_each_layer)):
        my_nodes_name_index_together.append(nods)
    my_nodes_name_index_layers = []
    start = 0
    end = number_of_nodes_on_each_layer[0]
    for i in range(len(number_of_nodes_on_each_layer)):
        my_nodes_name_index_layers.append(my_nodes_name_index_together[start: end])
        if i < len(number_of_nodes_on_each_layer)-1:
            start = start + number_of_nodes_on_each_layer[i]
            end = end + number_of_nodes_on_each_layer[i+1]
    print(my_nodes_name_index_together)
    print(my_nodes_name_index_layers)


    my_nodes_dict_together = {}
    for i in range(len(my_nodes_name_index_together)):
        my_nodes_dict_together[str(i)] = dict(color=str(my_nodes_name_index_together[i]), pos='')

    x_margin = 0.1
    y_margin = 0.1
    x_layer_step = 0.8/len(my_nodes_name_index_layers)
    y_node_step = []
    for i in range(len(my_nodes_name_index_layers)):
        if i == len(my_nodes_name_index_layers) - 1 and len(my_nodes_name_index_layers[i]) < len(my_nodes_name_index_layers[i-1]):
            inside = len(my_nodes_name_index_layers[i]) / len(my_nodes_name_index_layers[i - 1])
            y_node_step.append(inside / len(my_nodes_name_index_layers[i]))
            outside = 1 - inside
        else:
            y_node_step.append(1/len(my_nodes_name_index_layers[i]))
    print(x_layer_step, y_node_step)
    my_nodes_pos_dict = {}
    counter = 0
    for i in range(len(my_nodes_name_index_layers)):
        for j in range(len(my_nodes_name_index_layers[i])):
            if i == len(my_nodes_name_index_layers) - 1 and len(my_nodes_name_index_layers[i]) < len(my_nodes_name_index_layers[i-1]):
                if outside > 0:
                    my_nodes_pos_dict[str(counter)] = [i*x_layer_step + x_margin, j*y_node_step[i] + y_margin + outside/2]
                else:
                    my_nodes_pos_dict[str(counter)] = [i * x_layer_step + x_margin,
                                                       j * y_node_step[i] + y_margin]
            else:
                my_nodes_pos_dict[str(counter)] = [i * x_layer_step + x_margin, j * y_node_step[i] + y_margin]
            counter = counter + 1


    print("number of position coordinates: ", len(my_nodes_pos_dict), my_nodes_pos_dict)

    edges = []
    for i in range(len(my_nodes_name_index_layers)):
        if i < len(my_nodes_name_index_layers)-1:
            for j in range(len(my_nodes_name_index_layers[i])):
                for k in range(len(my_nodes_name_index_layers[i+1])):
                    edges.append(((str(my_nodes_name_index_layers[i][j])), str(my_nodes_name_index_layers[i+1][k])))
    print("my edges: ", len(edges), edges)
    #return Network_Graphize(number_of_nodes_on_each_layer, my_nodes_dict_together, my_nodes_name_index_together, edges, fig_width, fig_height)
    Network_Graphize(my_nodes_pos_dict, my_nodes_dict_together, my_nodes_name_index_together, edges,
                     fig_width, fig_height)

def Network_Graphize(my_nodes_pos_dict, my_nodes_dict_together, my_nodes_name_index_together, edges, fig_width, fig_height):
    network_graph = nx.Graph()

    network_graph.add_nodes_from([n for n in my_nodes_dict_together])
    network_graph.add_edges_from((u, v) for u, v in edges)

    posit = nx.spring_layout(network_graph, dim=2)

    for i in range(len(network_graph.nodes)):
        #print(network_graph.nodes[str(i)])
        network_graph.nodes[str(i)].update(pos=my_nodes_pos_dict[str(i)], color='')
        #print(network_graph.nodes[str(i)])

    edge_trace, edge_text_trace = Network_Edges_Interactive_Part(network_graph)
    node_trace = Network_Nodes_Interactive_Part(network_graph)

    node_trace, edge_trace, edge_text_trace = Network_Adjacency(network_graph, my_nodes_name_index_together, node_trace, edge_trace, edge_text_trace)
    #return node_trace, edge_trace,

    #fig = Figure_Drawing(edge_trace, edge_text_trace, node_trace, fig_width, fig_height)
    #return fig
    Figure_Drawing(edge_trace, edge_text_trace, node_trace, fig_width, fig_height)

def Network_Edges_Interactive_Part(network_graph):
    edge_x = []
    edge_y = []

    edge_x_marker = []
    edge_y_marker = []

    for edge in network_graph.edges():
        #print(edge)
        x0, y0 = network_graph.nodes[edge[0]]['pos']
        x1, y1 = network_graph.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_x_marker.append((x0 + x1) / 2)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_y_marker.append((y0 + y1) / 2)


    edge_trace = go.Scatter(x=edge_x,
                            y=edge_y,
                            #line=dict(width=5, color=[], colorscale=edge_colorscale),
                            line=dict(width=2, color="grey"),
                            hoverinfo='none',
                            mode='lines'
                            )
    edge_marker_trace = go.Scatter(x=edge_x_marker,
                                   y=edge_y_marker,
                                   mode='markers',
                                   hoverinfo='text',
                                   #marker_size=5,
                                   marker=dict(showscale=True, size=5, color='Red', #colorscale='rgb(0, 0, 255)',
                                               reversescale=False, line_width=0)
                                     )
    return edge_trace, edge_marker_trace


def Network_Nodes_Interactive_Part(network_graph):
    node_x = []
    node_y = []

    for node in network_graph.nodes():
        x, y = network_graph.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)



    node_trace = go.Scatter(x=node_x,
                            y=node_y,
                            mode='markers',
                            hoverinfo='text',
                            marker=dict(showscale=False,
                                        #colorscale='rgb(0, 255, 0)',
                                        reversescale=False,
                                        cmin=-0.5,
                                        cmax=6.5,
                                        color='rgb(255, 0, 0)',
                                        size=15)
                            )
    return node_trace




def Network_Adjacency(network_graph, my_nodes_name_index, node_trace, edge_trace, edge_text_trace):
    node_adjacencies = []
    node_text = []
    #node_names = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
    for node, adjacencies in enumerate(network_graph.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        #node_text.append(node_names[my_nodes_name_index[node]] + ' and # of connections: ' + str(len(adjacencies[1])))
        node_text.append("Node: "+ str(list(network_graph.nodes.keys())[node]))
    #print("nodes",list(original_graph.nodes.keys()))

    edge_text = []
    edge_type = []
    #bond_dict = {0: "aromatic", 1: "single", 2: "double", 3: "triple"}

    for i, edge in enumerate(network_graph.edges.values()):
        #edge_type.append(edge['type'])
        edge_text.append("edge")



    #print(my_nodes_name_index)
    #node_trace.marker.color = my_nodes_name_index
    node_trace.text = node_text

    #edge_trace.marker.color = edge_type
    #print(len(edge_type), edge_type)
    #edge_trace.line.color = edge_type
    #edge_text_trace.marker.color = edge_type
    edge_text_trace.text = edge_text
    #for i, edge in enumerate(original_graph.edges.values()):
    #    print(i, "  ", edge['type'])


    return node_trace, edge_trace, edge_text_trace



def Figure_Drawing(edge_trace, edge_text_trace, node_trace, fig_width, fig_height):
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )
    fig = go.Figure(data=[edge_trace, node_trace, edge_text_trace],
                    layout=go.Layout(plot_bgcolor='white',
                                     width=fig_width, height=fig_height,
                                     title='<br>Molecule',
                                     titlefont_size=16,
                                     showlegend=False,
                                     hovermode='closest',
                                     margin=dict(b=20, l=5, r=5, t=40),
                                     scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis)),
                                     annotations=[dict(showarrow=False,
                                                       xref="paper", yref="paper",
                                                       x=0, y=0, xanchor='right', yanchor='top')]
                                    )
                    )
    #fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.show()
    #return fig

Network_datastructure_part(1000, 700)
