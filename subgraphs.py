import torch
import torch_geometric
import math
from torch_geometric.datasets import TUDataset
import plotly.graph_objects as go
import networkx as nx
Mutag_Dataset = TUDataset(root='data/TUDataset', name='MUTAG')







def subgraph_datastructure(subgraph_index, sub_fig_width, sub_fig_height):

    g1_my_nodes_name_index = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    g2_my_nodes_name_index = [2, 2, 1, 0, 0, 0, 0, 0]
    g3_my_nodes_name_index = [5, 0, 0, 0, 0, 0, 0, 1]
    g4_my_nodes_name_index = [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1]


    g1_my_nodes_dict = {}
    for i in range(len(g1_my_nodes_name_index)):
        g1_my_nodes_dict[str(i)] = dict(color=str(g1_my_nodes_name_index[i]), pos='')

    g2_my_nodes_dict = {}
    for i in range(len(g2_my_nodes_name_index)):
        g2_my_nodes_dict[str(i)] = dict(color=str(g2_my_nodes_name_index[i]), pos='')

    g3_my_nodes_dict = {}
    for i in range(len(g3_my_nodes_name_index)):
        g3_my_nodes_dict[str(i)] = dict(color=str(g3_my_nodes_name_index[i]), pos='')

    g4_my_nodes_dict = {}
    for i in range(len(g4_my_nodes_name_index)):
        g4_my_nodes_dict[str(i)] = dict(color=str(g4_my_nodes_name_index[i]), pos='')

    g1_edges_indexes = [[0, 1, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 6, 6, 7, 7, 8, 7, 9],
                        [1, 0, 2, 1, 3, 1, 4, 2, 5, 3, 6, 4, 6, 5, 7, 6, 8, 7, 9, 7]]
    g2_edges_indexes = [[0, 2, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                        [2, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]]
    g3_edges_indexes = [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]]
    g4_edges_indexes = [[0, 3, 1, 3, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 8, 10],
                        [3, 0, 4, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 10, 8]]

    subgraph_index_dict = {0: [g1_my_nodes_name_index, g1_my_nodes_dict, g1_edges_indexes],
                           1: [g2_my_nodes_name_index, g2_my_nodes_dict, g2_edges_indexes],
                           2: [g3_my_nodes_name_index, g3_my_nodes_dict, g3_edges_indexes],
                           3: [g4_my_nodes_name_index, g4_my_nodes_dict, g4_edges_indexes]}

    return subgraph_graphize(subgraph_index_dict[subgraph_index], sub_fig_width, sub_fig_height)


def subgraph_graphize(input_graph_data, sub_fig_width, sub_fig_height):
    subgraph_my_nodes_name_index = input_graph_data[0]
    subgraph_my_nodes_dict = input_graph_data[1]
    subgraph_edges_indexes = input_graph_data[2]


    subgraph_edges = []
    for source, target in zip(subgraph_edges_indexes[0], subgraph_edges_indexes[1]):
        subgraph_edges.append((str(source), str(target)))

    #g2_edges = []
    #for i in range(len(g2_edges_indexes[0])):
    #    g2_edges.append((str(int(g2_edges_indexes[0][i])), str(int(g2_edges_indexes[1][i]))))

    #g3_edges = []
    #for i in range(len(g3_edges_indexes[0])):
    #    g3_edges.append((str(int(g3_edges_indexes[0][i])), str(int(g3_edges_indexes[1][i]))))

    #g4_edges = []
    #for i in range(len(g4_edges_indexes[0])):
    #    g4_edges.append((str(int(g4_edges_indexes[0][i])), str(int(g4_edges_indexes[1][i]))))

    subgraph_original = nx.Graph()
    subgraph_original.add_nodes_from([(n, {"type": atom_index}) for n, atom_index in zip(subgraph_my_nodes_dict, subgraph_my_nodes_name_index)])
    subgraph_original.add_edges_from((u, v) for u, v in subgraph_edges)

    #g2 = nx.Graph()
    #g2.add_nodes_from([(n, {"type": atom_index}) for n, atom_index in zip(g2_my_nodes_dict, g2_my_nodes_name_index)])
    #g2.add_edges_from((u, v) for u, v in g2_edges)

    #g3 = nx.Graph()
    #g3.add_nodes_from([(n, {"type": atom_index}) for n, atom_index in zip(g3_my_nodes_dict, g3_my_nodes_name_index)])
    #g3.add_edges_from((u, v) for u, v in g3_edges)

    #g4 = nx.Graph()
    #g4.add_nodes_from([(n, {"type": atom_index}) for n, atom_index in zip(g4_my_nodes_dict, g4_my_nodes_name_index)])
    #g4.add_edges_from((u, v) for u, v in g4_edges)


    subgraph_original_posit = nx.spring_layout(subgraph_original, dim=3)
    #g2_posit = nx.spring_layout(g2, dim=3)
    #g3_posit = nx.spring_layout(g3, dim=3)
    #g4_posit = nx.spring_layout(g4, dim=3)
    for i in range(len(subgraph_original.nodes)):
        subgraph_original.nodes[str(i)].update(pos=subgraph_original_posit[str(i)], color='')


    return update_subgraph(subgraph_original, subgraph_my_nodes_name_index, sub_fig_width, sub_fig_height)

def update_subgraph(subgraph_original, subgraph_my_nodes_name_index, sub_fig_width, sub_fig_height):
    subgraph_node_trace = subgraph_Nodes_Interactive_Part(subgraph_original)
    subgraph_node_trace, subgraph_graph_annotations = subgraph_Adjacency(subgraph_original, subgraph_my_nodes_name_index, subgraph_node_trace)

    subgraph_fig = subgraph_Figure_Drawing( subgraph_node_trace, sub_fig_width, sub_fig_height, subgraph_graph_annotations)
    return subgraph_fig






def subgraph_Edges_Interactive_Part(original_graph):
    edge_x = []
    edge_y = []
    edge_z = []
    edge_x_marker = []
    edge_y_marker = []
    edge_z_marker = []
    test_edge_x = []
    test_edge_y = []
    test_edge_z = []
    for edge in original_graph.edges():
        x0, y0, z0 = original_graph.nodes[edge[0]]['pos']
        x1, y1, z1 = original_graph.nodes[edge[1]]['pos']
        length = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)

        dotSizeConversion = .0565 / 20                                             # length units per node size
        convertedDotDiameter = 20 * dotSizeConversion
        lengthFracReduction = convertedDotDiameter / length
        lengthFrac = 1 - 0.20#lengthFracReduction

        skipX = (x1 - x0) * (1 - lengthFrac)
        skipY = (y1 - y0) * (1 - lengthFrac)
        skipZ = (z1 - z0) * (1 - lengthFrac)

        x0 = x0 + skipX / 2
        x1 = x1 - skipX / 2
        y0 = y0 + skipY / 2
        y1 = y1 - skipY / 2
        z0 = z0 + skipZ / 2
        z1 = z1 - skipZ / 2

        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_x_marker.append((x0 + x1) / 2)

        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_y_marker.append((y0 + y1) / 2)

        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)
        edge_z_marker.append((z0 + z1) / 2)


        test_edge_x.append([x0, x1, None])
        test_edge_y.append([y0, y1, None])
        test_edge_z.append([z0, z1, None])

    bond_color_dict = {0: 'rgb(255, 0, 0)', 1: 'rgb(0, 255, 0)', 2: 'rgb(210,105,30)', 3: 'rgb(255, 255, 0)'}
    edge_colors = []
    for i, edge in enumerate(original_graph.edges.values()):
        edge_colors.append(bond_color_dict[edge['type']])

    #colors = ['red'] * len(original_graph.edges)
    link_size = [5] * len(original_graph.edges)
    traces_for_edges = []



    for i in range(len(original_graph.edges)):
        traces_for_edges.append(go.Scatter3d(x=test_edge_x[i],
                                             y=test_edge_y[i],
                                             z=test_edge_z[i],
                                             hoverinfo='none',
                                             line=dict(color=edge_colors[i], width=link_size[i]),
                                             mode='lines'
                                             )
                                )






    return traces_for_edges





def subgraph_Nodes_Interactive_Part(original_graph):
    node_x = []
    node_y = []
    node_z = []
    for node in original_graph.nodes():
        x, y, z = original_graph.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    # node_names = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
    node_names = ['Carbon', 'Nitrogen', 'Oxygen', 'Fluorine', 'Indium', 'Chlorine', 'Bromine']
    node_color_vals = list(range(len(node_names)))
    num_colors = len(node_color_vals)
    node_colorscale = [[0, 'rgb(255, 0, 0)'], [0.14285714, 'rgb(255, 0, 0)'], # red
                       [0.14285714, 'rgb(0, 255, 0)'], [0.28571429, 'rgb(0, 255, 0)'],
                       [0.28571429, 'rgb(191, 62, 255)'], [0.42857143, 'rgb(191, 62, 255)'],
                       [0.42857143, 'rgb(255, 127, 0)'], [0.57142857, 'rgb(255, 127, 0)'],
                       [0.57142857, 'rgb(0, 255, 255)'], [0.71428571, 'rgb(0, 255, 255)'],
                       [0.71428571, 'rgb(255, 28, 174)'], [0.85714286, 'rgb(255, 28, 174)'],
                       [0.85714286, 'rgb(255, 255, 0)'], [1.0, 'rgb(255, 255, 0)']
                       ]
    node_trace = go.Scatter3d(x=node_x,
                              y=node_y,
                              z=node_z,
                              mode='markers',
                              hoverinfo='text',
                              marker=dict(colorscale=node_colorscale,
                                          color=[],
                                          showscale=True,
                                          reversescale=False,
                                          cmin=-0.5,
                                          cmax=6.5,
                                          #color='white',
                                          size=20,
                                          sizemode='diameter',
                                          symbol=[],
                                          colorbar=dict(thickness=15, tickvals=node_color_vals,
                                                        ticktext=node_names,
                                                        orientation='h',
                                                        title='Atoms',
                                                        xanchor='center',
                                                        titleside='top'
                                                        ),
                                          #line=dict(color=["black"], width=[8]),
                                          #line_width=0,
                                          )
                              )
    return node_trace










def subgraph_Adjacency(original_graph, my_nodes_name_index, node_trace):
    node_adjacencies = []
    node_text = []
    node_names = {0: 'Carbon', 1: 'Nitrogen', 2: 'Oxygen', 3: 'Fluorine', 4: 'Indium', 5: 'Chlorine', 6: 'Bromine'}
    for node, adjacencies in enumerate(original_graph.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(node_names[my_nodes_name_index[node]] + ' <br># of connections: ' + str(len(adjacencies[1])))



    node_trace.marker.color = my_nodes_name_index
    #node_trace.marker.line.color = my_nodes_name_index
    node_trace.text = node_text
    node_trace.marker.symbol = ['circle-open'] * (len(original_graph.nodes))




    ##################################################################     Annotations     #############################
    nick_node_names = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}

    annotations_original_graph = []
    for i in range(len(original_graph.nodes)):
        posit = original_graph.nodes[str(i)]['pos']
        annotations_original_graph.append(dict(x=posit[0], y=posit[1], z=posit[2],
                                               text=nick_node_names[original_graph.nodes[str(i)]['type']], xanchor='left',
                                               xshift=-5, yshift=+1, font=dict(color='black', size=12),
                                               showarrow=False, arrowhead=1, ax=0, ay=0
                                               )
                                          )



    return node_trace, annotations_original_graph





def subgraph_Figure_Drawing( node_trace, fig_width, fig_height, original_graph_annotations):
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )

    Data = [node_trace]


    fig = go.Figure(data=Data,
                    layout=go.Layout(plot_bgcolor='white',
                                     width=fig_width, height=fig_height,
                                     title=dict(text="<b>Subgraph</b> Label: ", y=0.96, xanchor='left', yanchor='top'),
                                     titlefont_size=16,
                                     showlegend=False,
                                     hovermode='closest',
                                     margin=dict(b=20, l=5, r=5, t=40),
                                     scene=dict(annotations=original_graph_annotations, xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis)),
                                     #annotations=[dict(showarrow=False,
                                     #                  xref="paper", yref="paper",
                                     #                  x=0, y=0, xanchor='right', yanchor='top')]
                                    )
                    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig


print(subgraph_datastructure(0, 200, 200))

#fig = Figure_Drawing(Graph, edge_marker_trace, node_trace, traces_for_edges, fig_width, fig_height, original_graph_annotations)


#fig_width=200
#fig_height=200
#fig = subgraph_Figure_Drawing(g1_node_trace, fig_width, fig_height, g1_graph_annotations)




#print(g1.nodes)

#for i in range(len(g1.nodes)):
#    g1.nodes[str(i)].update(pos=g1_posit[str(i)], color='')


#for i in range(len(g2.nodes)):
#    g2.nodes[str(i)].update(pos=g2_posit[str(i)], color='')


#for i in range(len(g3.nodes)):
#    g3.nodes[str(i)].update(pos=g3_posit[str(i)], color='')


#for i in range(len(g4.nodes)):
#    g4.nodes[str(i)].update(pos=g4_posit[str(i)], color='')




#print("g1 nodes", g1_my_nodes_dict)
#print(g1.nodes)
#print("g2 nodes", g2_my_nodes_dict)
#print(g2.nodes)
#print("g3 nodes", g3_my_nodes_dict)
#print(g3.nodes)
#print("g4 nodes", g4_my_nodes_dict)
#print(g4.nodes)


#g1_node_trace = subgraph_Nodes_Interactive_Part(g1)
#g1_node_trace, g1_graph_annotations = subgraph_Adjacency(g1, g1_my_nodes_name_index, g1_node_trace)






#print(Mutag_Dataset[0].edge_index.detach().tolist()[0])
#print(Mutag_Dataset[0].edge_index.detach().tolist()[1])