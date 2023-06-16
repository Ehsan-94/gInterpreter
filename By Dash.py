import os
import random
from typing import Any, List, Dict, Tuple, Union, Callable
from collections import defaultdict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import tkinter as tk
from tkinter import messagebox
import torch
import torch_geometric
from torch_geometric.datasets import TUDataset
import pyvis
from pyvis.network import Network
from math import sin, cos
import networkx as nx
from networkx.algorithms import isomorphism
from pyvis.network import Network
from urllib.request import urlopen
#import tkinterhtml as th
#import webview
#from tkinterweb import HtmlFrame
import urllib
import torch.nn.functional as F
import dash
from dash import Dash, dcc, html, Input, Output, State, dash_table
from dash.dash_table.Format import Group, Format, Scheme
import plotly
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import dash_bootstrap_components as dbc
import dash_daq as daq
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
#from d3graph import d3graph, vec2adjmat
#from matplotlib.figure import Figure
#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#import dash_cytoscape as cyto
import Model_Loading as GCN_Model
import GCN_plus_GAP as Graph_Network
from operator import add
import math
import pandas
from ast import literal_eval
from time import perf_counter
###################################################################################################    Methods    ######
import SA_Method_as_Class
import GuidedBP_Method_as_Class
import CAM_Method_as_Class
import Grad_CAM_Method_as_Class
import LRP_Method_as_Class
import ExcitationBP_Method_as_Class
import GNNExplainer_Method_as_Class
import PGExplainer_Method_as_Class
import GraphMask_Method_as_Class
import SubGraphX_offline_Method_as_Class
import PGMExplainer_Method_as_Class
import CF2Explainer_Method_as_Class
from dash_daq import Slider





#Mutag_Dataset = shuffle(TUDataset(root='data/TUDataset', name='MUTAG'))
Mutag_Dataset = TUDataset(root='data/TUDataset', name='MUTAG')

def load_nerd_model(model_level, dim_node, dim_hidden, dim_output):
    classifier_act_fun = 'eLu'
    model_name = 'GCN_plus_GAP'
    classifier_dropout_rate = 0.1
    GNN_Model = Graph_Network.GCN_plus_GAP(model_name=model_name, model_level=model_level, input_dim=dim_node, hidden_dim=dim_hidden,
                                           output_dim=dim_output, num_hid_layers=2, Bias=True, act_fun=classifier_act_fun,
                                           Weight_Initializer=1, dropout_rate=classifier_dropout_rate)

    # GNN_Model = GCN_Model.GCN_2Layer_Model(model_level, dim_node=dim_node, dim_hidden=dim_hidden, dim_output=dim_output)
    return GNN_Model

def loading_pretrained_model(load_index, Explainability_name, Model_name, Task_name):
    #print("Loading Pretrained Model", load_index, Explainability_name, Task_name)
    DataSet_name = "MUTAG"
    if Explainability_name == "CF\u00b2":
        Explainability_name = "CF2"
    GNN_model = load_nerd_model('graph', 7, 7, 2)
    optimizer = torch.optim.Adam(GNN_model.parameters(), lr=0.001, weight_decay=1e-6)
    checkpoint = torch.load(str(Model_name) + " " + str(Explainability_name) + " " + str(Task_name) + " " + str(DataSet_name) + " " + str(load_index)+".pt")
    GNN_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']


    return GNN_model, optimizer, epoch

def loading_test_samples():
    df = pandas.read_csv("/Users/EY33JW/PycharmProjects/pythonProject2/train_test_indexes.csv")

    read_training_list_indexes__ = df['Train Indexes']
    read_test_list_indexes__ = df['Test Indexes']
    read_test_list_indexes__ = read_test_list_indexes__.dropna()
    read_test_list_indexes = []
    read_training_list_indexes = []
    for element in read_test_list_indexes__:
        read_test_list_indexes.append(int(element))
    for element in read_training_list_indexes__:
        read_training_list_indexes.append(int(element))

    test_samples = []
    for index in read_test_list_indexes:
        test_samples.append(Mutag_Dataset[index])


    #print(read_training_list_indexes)
    #print(read_test_list_indexes)
    return test_samples, read_training_list_indexes, read_test_list_indexes



#print(load_nerd_model('graph', 7, 7, 2))
#my_model = load_nerd_model('graph', 7, 7, 2)
#out_not_pretrained = my_model(Mutag_Dataset[0].x, Mutag_Dataset[0].edge_index)

#Explainability_name = 'SA'
#Task_name = 'Graph Classification'
#load_index = 200
#my_model_pretrained, optimizer, epochs = loading_pretrained_model(load_index, Explainability_name, Task_name)
#out_pretrained = my_model_pretrained(Mutag_Dataset[0].x, Mutag_Dataset[0].edge_index)

#print(out_not_pretrained)
#print(out_pretrained)

#Node_Counts_on_Dataset = []
#Edge_Counts_on_Dataset = []
#for graph in Mutag_Dataset:
#    Node_Counts_on_Dataset.append(len(graph.x))
#    Edge_Counts_on_Dataset.append(len(graph.edge_index[0]))


#print(sum(Node_Counts_on_Dataset))
#print(sum(Edge_Counts_on_Dataset))
#mut_edge = np.loadtxt('MUTAG_edge_labels.txt')
#mut_node = np.loadtxt('MUTAG_node_labels.txt')
#A= np.loadtxt('MUTAG_A.txt', dtype=str)


#color_dict= {0:"Red", 1:"Blue", 2:"Yellow",3:"Green", 4:"Gray", 5:"Purple", 6:"Pink"}
#node={}
#for i in range(sum(Node_Counts_on_Dataset)):
#   node[str(i)]=color_dict[mut_node[i]]

#bond_dict = {0:"aromatic", 1:"single", 2:"double", 3:"triple"}
#edges=[]
#for i in range(sum(Edge_Counts_on_Dataset)):
#   edges.append((A[i][0].split(',')[0], A[i][1], bond_dict[mut_edge[i]]))
fig_width = 1530
fig_height = 693

net_fig_width = 1530
net_fig_height = 467

isomer_fig_width = 400
isomer_fig_height = 300

subgraph_fig_width = 400
subgraph_fig_height = 300
#importance_threshold = 0.5
Graphs_Family, training_indexes_read, test_indexes_read = loading_test_samples()



dropdowns_width = '400px'
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load dataset using Plotly

controls = dbc.Card(
    [
#        dbc.CardHeader("A web application framework for your data", style={'text-align': 'center', 'padding': '0px',
#                                                                           "font-weight": "bold", "height": "23px",
#                                                                           'backgroundColor': '#e0e0e0',
#                                                                           'margin-bottom': '2px',
#                                                                           'font-family': 'Times New Roman'}),
        #html.Div(id='GUI_Input', children=[
            #html.H1(children='GUI for XAI on Graph-Structured Data',
            #        style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 40, 'color': 'red'}),
            # Create a title with H1 tag

            html.Div(children='''''',
                     style={'textAlign': 'left', 'color': 'grey'}),  # Display some text
            dcc.Dropdown(id='Task_Dropdown',
                         options=[{'label': 'Graph Classification', 'value': 'GC'},
                                  {'label': 'Node Classification', 'value': 'NC'},
                                  {'label': 'Link Prediction', 'value': 'LP'},
                                  {'label': 'Entity Resolution', 'value': 'ER'}],
                         value='',
                         placeholder='Please Select a Task...',
                         style={'width': dropdowns_width, 'margin-bottom': '2px', 'margin-top': '2px','font-family': 'Times New Roman'}),
            dcc.Dropdown(id='Index_Dropdown',
                         options=[],
                         value='',
                         placeholder='Please Select The Task at First...',
                         style={'width': dropdowns_width, 'margin-bottom': '2px', 'font-family': 'Times New Roman'}),
            dcc.Dropdown(id='Model_Dropdown',
                         options=[],
                         value='',
                         placeholder='Please Select The Task at First...',
                         style={'width': dropdowns_width, 'margin-bottom': '2px', 'font-family': 'Times New Roman'}),
            dcc.Dropdown(id='Method_Dropdown',
                         options=[],
                         value='',
                         placeholder='Please Select The Task at First...',
                         style={'width': dropdowns_width, 'margin-bottom': '2px', 'font-family': 'Times New Roman'}),
            html.Button(id='Start_Button', n_clicks=0, children="Click to Start",
                        style={'font-size': '20px', 'color': "success", 'width': dropdowns_width, 'height': '35px',
                               'display': 'inline-block', 'font-family': 'Times New Roman',
                               'margin-up': 'auto', 'c-button': 'normal', 'margin-right': 'auto',
                               'background-color': '#949494', 'margin-bottom': '2px',
                               'border-color': '#949494', 'class': 'pull-left', 'color': 'white', 'border-radius': '8px'},
                        type='success'),
        #]),
    ], style={'margin-bottom': '2px', 'margin-top': '2px', 'background-color': '#c0c0c0'}, body=True,
)
display_toggle_and_slider_controls = dbc.Card(
    [
        daq.BooleanSwitch(id="my_display_toggle", on=False, color="red", labelPosition='right',
                          style={'font-family': 'Times New Roman', 'margin-top': '10px',
                                 'margin-left': '10px', 'margin-right': '10px'},
                          label={"label": "Graph Display",
                                 'style': {"width": "300px", 'margin-left': '10px', 'margin-top': '0px'}}),

        daq.BooleanSwitch(id="my_network_display_toggle", labelPosition='right', on=False, color="red",
                          style={'font-family': 'Times New Roman', 'margin-top': '10px',
                                 'margin-left': '10px', 'margin-right': '10px'},
                          label={"label": "Neural Network Display",
                                 'style': {"width": "300px", 'margin-left': '10px', 'margin-top': '0px'}}),

        daq.BooleanSwitch(id="my_model_and_method_general_statistics_display_toggle", labelPosition='right', on=False, color="red",
                          style={'font-family': 'Times New Roman', 'margin-top': '10px',
                                 'margin-left': '10px', 'margin-right': '10px'},
                          label={"label": "Model and Method General Performance Display",
                                 'style': {"width": "300px", 'margin-left': '10px', 'margin-top': '0px'}}),

        daq.BooleanSwitch(id="my_model_and_method_instance_specific_statistics_display_toggle", labelPosition='right', on=False, color="red",
                          style={'font-family': 'Times New Roman', 'margin-top': '10px',
                                 'margin-left': '10px', 'margin-right': '10px', 'margin-bottom': '50px'},
                          label={"label": "Model and Method Instance-Specific Performance Display",
                                 'style': {"width": "300px", 'margin-left': '10px', 'margin-top': '0px'}}),
        html.Div(daq.Slider(id='importance_threshold_my-slider', value=0.5, min=0, max=1, step=0.01, marks=None,
                            handleLabel={"showCurrentValue": True, "label": "Threshold"}, color="red"),
                 style={'margin-left': '0px', 'margin-bottom': '10px', 'padding-top': '0px', 'padding-left': '65px',
                        'padding-right': '10px', 'padding-bottom': '0px'}),

    ], body=True, style={'margin-right': '0px', 'background-color': '#c0c0c0'}
)


display_screen = dbc.Card(
    [
        html.Div([dcc.Graph(id="plot")], id="Visualize_Graph", style={"width": "100%", "height": "100%",
                                                                      'padding': '0px', 'margin-bottom': '0px'})
    ],
    body=True, style={'margin-bottom': '0px', 'padding': '0px', 'background-color': '#D3D3D3'}
)


display_network = dbc.Card(
    [
        html.Div([dcc.Graph(id="plot_net")], id="Visualize_Model", style={"width": "100%", "height": "100%",
                                                                          'margin-left': '0px', 'padding': '0px'})
    ],
    body=True, style={'margin-top': '0px', 'padding': '0px', 'background-color': '#D3D3D3'}
)

display_classifier_generally = dbc.Card(
    [
        dbc.CardHeader("Classifier General Performance", style={'text-align': 'center', "font-weight": "bold",
                                                                'padding': '0px', "height": "23px", 'font-size': '18px',
                                                                'backgroundColor': '#e0e0e0', 'margin-top':'0px',
                                                                'font-family': 'Times New Roman'}),
        dash_table.DataTable(id='Visualize_General_Performance_of_Classifier', style_as_list_view=True,
                             #style_table={'overflowX': 'auto', 'textOverflow': 'ellipsis'},
                             #style_data={'whiteSpace': 'normal',
                             #            'width': '10px'
                              #           },
                             style_cell={'textOverflow': 'ellipsis',
                                         'overflowX': 'auto',
                                         'textAlign': 'center',
                                         'textOverflow': 'ellipsis',
                                         'maxWidth': '20px',
                                         'backgroundColor': '#f9f9f9',
                                         },
                             style_header={'textAlign': 'center', 'white-space': 'initial',#'textOverflow': 'ellipsis',
                                           'backgroundColor': '#ededed', 'fontWeight': 'bold', 'margin-top': '5px',
                                           }
                             )
        #html.Div([dcc.Graph (id="plot_performance")], id="Visualize_Performance", style={"width": "100%", "height": "100%"})
    ],
    body=True, style={'padding': '0px', 'margin-right': '0px', 'background-color': '#c0c0c0'}
)
display_explainer_generally = dbc.Card(
    [
        dbc.CardHeader("Explainer General Performace", style={'text-align': 'center', "font-weight": "bold",
                                                              'padding': '0px', "height": "23px",
                                                              'backgroundColor': '#e0e0e0', 'font-size': '18px',
                                                              'font-family': 'Times New Roman'}),
        dash_table.DataTable(id='Visualize_General_Performance_of_Explainer', style_as_list_view=True,
                             style_cell={'textOverflow': 'ellipsis',
                                         'overflowX': 'auto',
                                         'textAlign': 'center',
                                         'textOverflow': 'ellipsis',
                                         'maxWidth': '20px',
                                         'backgroundColor': '#f9f9f9',
                                         },
                             style_header={'textAlign': 'center', 'white-space': 'initial', #'textOverflow': 'ellipsis',
                                           'backgroundColor': '#ededed', 'fontWeight': 'bold'}
                             )
        #html.Div([dcc.Graph(id="plot_performance")], id="Visualize_Performance", style={"width": "100%", "height": "100%"})
    ],
    body=True, style={'padding': '0px', 'margin-right': '0px', 'background-color': '#c0c0c0'}
)

display_classifier_instance_specific = dbc.Card(
    [
        dbc.CardHeader("Classifier Instance-Specific", style={'text-align': 'center', "font-weight": "bold",
                                                              'padding': '0px', "height": "23px",
                                                              'backgroundColor': '#e0e0e0', 'font-size': '18px',
                                                              'font-family': 'Times New Roman'}),
        dash_table.DataTable(id='Visualize_Instance_Specific_Performance_of_Classifier', style_as_list_view=True,
                             style_cell={'textOverflow': 'ellipsis',
                                         'overflowX': 'auto',
                                         'textAlign': 'center',
                                         #'textOverflow': 'ellipsis',
                                         'maxWidth': '20px',
                                         'backgroundColor': '#f9f9f9',
                                         },
                             style_header={'textAlign': 'center', 'white-space': 'initial',#'textOverflow': 'ellipsis',
                                           'backgroundColor': '#ededed', 'fontWeight': 'bold'
                                           }
                             )
        #html.Div([dcc.Graph (id="plot_performance")], id="Visualize_Performance", style={"width": "100%", "height": "100%"})
    ],
    body=True, style={'padding': '0px', 'margin-right': '0px', 'margin-left': '0px', 'background-color': '#c0c0c0'}
)

display_explainer_instance_specific = dbc.Card(
    [
        dbc.CardHeader("Explainer Instance-Specific", style={'text-align': 'center', "font-weight": "bold",
                                                             'padding': '0px', "height": "23px", 'font-size': '18px',
                                                             'backgroundColor': '#e0e0e0', 'font-family': 'Times New Roman'}),
        #html.Label("Explainer Instance-Specific", style={'display': 'flex', 'align': 'center'}),  #   'text-align': 'center'
        dash_table.DataTable(id='Visualize_Instance_Specific_Performance_of_Explainer', style_as_list_view=True,
                             style_cell={'textOverflow': 'ellipsis',
                                         'overflowX': 'auto',
                                         'textAlign': 'center',
                                         #'textOverflow': 'ellipsis',
                                         'maxWidth': '20px',
                                         'backgroundColor': '#f9f9f9',
                                         },
                             style_header={'textAlign': 'center', 'white-space': 'initial',#'textOverflow': 'ellipsis',
                                           'backgroundColor': '#ededed', 'fontWeight': 'bold'
                                           }
                             )
        #html.Div([dcc.Graph (id="plot_performance")], id="Visualize_Performance", style={"width": "100%", "height": "100%"})
    ],
    body=True, style={'padding': '0px', 'margin-right': '0px', 'margin-left': '0px', 'background-color': '#c0c0c0'}
)

display_isomer_graph = dbc.Card(
    [
        html.Div([dcc.Graph(id="plot_isomer")], id="Visualize_iSomer_Graph", style={"width": "100%", "height": "100%",
                                                                                    'margin-right': '0px',
                                                                                    'margin-left': '0px',})
    ],
    body=True, style={'margin-right': '0px', 'margin-left': '0px', 'padding': '0px', 'background-color': '#c0c0c0'}
)


display_subgraph_graph = dbc.Card(
    [
        html.Div([dcc.Graph(id="plot_subgraph")], id="Visualize_Subgraph_Graph", style={"width": "100%", "height": "100%",
                                                                                        'margin-right': '0px',
                                                                                        'margin-left': '0px',
                                                                                        'margin-bottom': '0px',})
    ],
    body=True, style={'margin-right': '0px', 'margin-left': '0px', 'padding': '0px', 'margin-bottom': '0px',
                      'background-color': '#c0c0c0'}
)

display_dropdown_button_isomer_graph = dcc.Dropdown(id='iSomer_Dropdown', options=[], value='', placeholder='iSomers',
                                                    style={'height': '38px', 'width': "130px", 'margin-top': '3px',
                                                           'font-family': 'Times New Roman', 'margin-right': '4px',
                                                           'font-size': '12px'})

display_dropdown_button_subgraphs = dcc.Dropdown(id='Subgraph_Dropdown', options=[], value='', placeholder='Subgraphs',
                                                    style={'height': '38px', 'width': "200px", 'margin-top': '3px',
                                                           'font-family': 'Times New Roman', 'font-size': '12px',
                                                           'margin-right': '4px'},
                                                 optionHeight=50)



button_download_pretrained_model_weights = html.Button(id='Download_Button', n_clicks=0, children="Download Weights",
                                                       style={
                                                           'font-size': '12px', 'color': "grey", 'width': '133px',
                                                           'height': '41px', 'display': 'inline-block',
                                                           'font-family': 'Times New Roman', 'margin-up': 'auto',
                                                           'c-button': 'normal', 'background-color': '#949494',
                                                           'margin-right': '0px', 'margin-left': '11px',
                                                           'margin-bottom': '0px', 'margin-top': '3px',
                                                           'border-color': '#a7aeb4', 'class': 'pull-left',
                                                           'color': 'white', 'border-radius': '8px'},
                                                       type='success')

dowload_process_pretrained_model_weights = dcc.Download(id="download_your_pretrained_model_weights")
left_margin = 5
right_margin = 5

app.layout = dbc.Container(
    [
        html.H1("GUI for XAI on Graph-Structured Data", style={'textAlign': 'center', 'marginTop': 10,
                                                               'marginBottom': 10, 'color': 'grey'}),
        html.Hr(),
        dbc.Stack(
                [
                dbc.Row(    ############  1.1 Row
                    [
                        dbc.Stack(
                            [
                                dbc.Col(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Stack(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Card(
                                                                    [
                                                                        dbc.CardHeader("A web application framework for your data",
                                                                                       style={'text-align': 'center', 'padding': '0px',
                                                                                              "font-weight": "bold", "height": "23px",
                                                                                              'backgroundColor': '#e0e0e0',
                                                                                              'margin-bottom': '0px',
                                                                                              'font-family': 'Times New Roman'}),
                                                                        controls,

                                                                    ], body=True, style={'margin-top': '0px',
                                                                                         'background-color': '#D3D3D3'}),

                                                                        #dbc.CardHeader("Select iSomer/s",
                                                                         #              style={'text-align': 'center', 'padding': '0px',
                                                                          #                    "font-weight": "bold", "height": "23px",
                                                                           #                   'backgroundColor': '#e0e0e0',
                                                                            #                  'margin-bottom': '0px',
                                                                             #                 'font-family': 'Times New Roman'}),
                                                                dbc.Row(
                                                                    [
                                                                        display_dropdown_button_isomer_graph,
                                                                        display_dropdown_button_subgraphs,
                                                                        button_download_pretrained_model_weights,
                                                                        dowload_process_pretrained_model_weights
                                                                    ], align='start', style={'margin-top': '0px'}
                                                                )
                                                                     #style={'margin-right': '0px', 'margin-left': '0px',
                                                                        #                 'margin-bottom': '0px', 'margin-top': '5px',
                                                                         #                'padding': '0px'}


                                                            ], md=6, width=0, style={"height": "100%", 'margin-right': '0px', 'margin-left': '0px',
                                                                                     'margin-bottom': '0px', 'margin-top': '0px'},
                                                            align="start",),
                                                        dbc.Col(
                                                            [
                                                                dbc.Card(
                                                                    [
                                                                        dbc.CardHeader("Graphical Controls",
                                                                                       style={'text-align': 'center', 'padding': '0px',
                                                                                              "font-weight": "bold", "height": "23px",
                                                                                              'backgroundColor': '#e0e0e0',
                                                                                              'margin-bottom': '2px',
                                                                                              'font-family': 'Times New Roman',}),
                                                                        display_toggle_and_slider_controls
                                                                        #html.Div("Second Column")
                                                                    ], body=True, style={'margin-top': '0px', 'background-color': '#D3D3D3'})
                                                            ], md=6, width=2, style={'margin-right': '0px', 'margin-left': '0px',
                                                                                     'margin-bottom': '5px', 'margin-top': '0px'},
                                                            align="start",)
                                                        ], gap=1, direction="horizontal"
                                                )
                                                #dbc.Row(dbc.Card([html.Div("second row")], body=True), align="start")
                                            ], align="start", style={'margin-bottom': '0px', 'margin-top': '0px'}
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Stack(
                                                    [
                                                    dbc.Col(
                                                        [
                                                            dbc.Card(
                                                                [
                                                                    dbc.CardHeader("iSomer Display",
                                                                                   style={'text-align': 'center', 'padding': '0px',
                                                                                          "font-weight": "bold", "height": "23px",
                                                                                          'backgroundColor': '#e0e0e0', 'margin-bottom': '0px',
                                                                                          'font-family': 'Times New Roman'
                                                                                          }
                                                                                   ),
                                                                    display_isomer_graph,
                                                                ], body=True, style={'margin-right': '0px', 'margin-left': '0px',
                                                                                     'padding': '0px', 'background-color': '#D3D3D3'}
                                                            )
                                                        ], align="start", md=6, width=0, style={'margin-right': '0px',
                                                                                                'margin-left': '0px',}
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Card(
                                                                [
                                                                    dbc.CardHeader("Subgraphs Display",
                                                                                   style={'text-align': 'center', 'padding': '0px',
                                                                                          "font-weight": "bold", "height": "23px",
                                                                                          'backgroundColor': '#e0e0e0',
                                                                                          'margin-bottom': '0px',
                                                                                          'font-family': 'Times New Roman'
                                                                                          }
                                                                                   ),
                                                                    #html.Div("second column for subgraphs")
                                                                    display_subgraph_graph
                                                                ], body=True, style={'margin-right': '0px', 'margin-left': '0px',
                                                                                     'margin-bottom': '0px', 'background-color': '#D3D3D3'}
                                                            )
                                                        ], align="start", md=6, width=0, style={'margin-right': '0px',
                                                                                                'margin-left': '0px',
                                                                                                'margin-bottom': '0px',
                                                                                                'margin-top': '0px',
                                                                                                'padding': '0px',
                                                                                                'height': '100%'},
                                                    ),
                                                    ], gap=1, direction="horizontal", style={'margin-bottom': '0px'},
                                                )
                                            ], align='start',
                                        )    #    row ends ghere
                                    ], align='start', md=4.5, width=0, style={'margin-right': '0px'}
                                ),
                                dbc.Col(display_screen, md=7.5, align="start", width=0, style={'margin-bottom': '0px', 'padding': '0px'}),
                            ], gap=2, direction='horizontal', style={'margin-bottom': '0px', 'height': '100%'},
                        ),
                    ], align="start", style={'margin-right': right_margin, 'margin-left': left_margin, 'margin-bottom': '0px'}
                ),
                #html.H2(style={'marginTop': 2, 'marginBottom': 2}),
                dbc.Row(
                    [
                    dbc.Stack(
                        [
                            #dbc.Col(display_classifier_generally, md=2),
                            ### dbc.Col(html.Div("middle part"), md=2),
                            #dbc.Col(display_explainer_generally, md=2),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.Stack(
                                            [
                                                dbc.Row(display_classifier_generally, style={"maxWidth": "950px", 'margin-top': '0px', 'margin-left': '0px', 'margin-right': '0px', 'padding': '0px'}),
                                                dbc.Row(display_explainer_generally, style={"maxWidth": "950px", 'margin-top': '0px', 'margin-left': '0px', 'margin-right': '0px', 'padding': '0px'}),
                                                dbc.Row(
                                                    [
                                                        dbc.Col([display_classifier_instance_specific], style={"maxWidth": "700px",
                                                                                                               'width': '100%',
                                                                                                               'margin-top': '0px',
                                                                                                               'margin-right': '0px',
                                                                                                               'margin-left': '0px',
                                                                                                               #'padding': '0px',
                                                                                                               'padding': '0px'}),
                                                        dbc.Col([display_explainer_instance_specific], style={"maxWidth": "250px",
                                                                                                              'width': '100%',
                                                                                                              'margin-top': '0px',
                                                                                                              'margin-left': '4px',
                                                                                                              'margin-right': '0px',
                                                                                                              'padding': '0px'}),
                                                    ], align='start', style={'margin-top': '0px', 'margin-bottom': '0px',
                                                                             'margin-right': '0px', 'margin-left': '0px'}
                                                ),
                                            ], gap=1, direction="vertical", style={'margin-top': '0px', 'width': '100%',
                                                                                   'margin-right': '0px', 'padding': '0px'}  #interesting
                                        ),
                                        #dbc.Row(),
                                    ], body=True, style={'margin-right': '0px', 'margin-left': '0px',
                                                         'margin-top': '0px', 'width': '100%', 'background-color': '#D3D3D3'}
                                ), align="start", style={'margin-left': '12px', 'margin-top': '0px', 'margin-bottom': '0px',
                                                               'margin-right': '0px', 'padding': '0px', "maxWidth": "944px"}
                            ),
                            dbc.Col(display_network, md=7.5, width=0, style={'margin-right': '0px', 'margin-left': '0px',
                                                                             'margin-bottom': '0px', 'padding': '0px'}),
                        ], gap=1, direction='horizontal', style={'margin-top': '0px', 'margin-right': '0px', 'width': '100%',
                                                                 'margin-left': '0px', 'padding': '0px'}
                    ),
                    ], align="start", style={'margin-top': '0px', 'margin-bottom': '0px', 'margin-right': right_margin, 'margin-left': left_margin}
                ),
                ], gap=1, direction='vertical', style={'margin-top': '0px', 'margin-left': '0px', 'padding': '0px',
                                                       'margin-right': '0px', 'margin-bottom': '0px'}
        )
    ], fluid=True, style={'margin-right': '0px', 'margin-left': '0px'}
)








@app.callback([Output(component_id='Index_Dropdown', component_property='value'),
               Output(component_id='Index_Dropdown', component_property='placeholder'),
               Output(component_id='Index_Dropdown', component_property='options'),

               Output(component_id='Model_Dropdown', component_property='value'),
               Output(component_id='Model_Dropdown', component_property='placeholder'),
               Output(component_id='Model_Dropdown', component_property='options'),

               Output(component_id='Method_Dropdown', component_property='value'),
               Output(component_id='Method_Dropdown', component_property='placeholder'),
               Output(component_id='Method_Dropdown', component_property='options')],

              [Input(component_id='Task_Dropdown', component_property='value')])
def graph_DropDowns(Task_Dropdown):
    print(Task_Dropdown)
    if Task_Dropdown == "GC":
        return "Please select the Graph", "Please select the Graph", [{'label': "Graph One", 'value': 0}, {'label': "Graph Two", 'value': 1}, {'label': "Graph Three", 'value': 2}, {'label': "Graph Four", 'value': 3}, {'label': "Graph Five", 'value': 4}, {'label': "Graph Six", 'value': 5}, {'label': "Graph Seven", 'value': 6}, {'label': "Graph Eight", 'value': 7}, {'label': "Graph Nine", 'value': 8}, {'label': "Graph Ten", 'value': 9}, {'label': "Graph Eleven", 'value': 10}, {'label': "Graph Twelve", 'value': 11}, {'label': "Graph Thirteen", 'value': 12}, {'label': "Graph Fourteen", 'value': 13}, {'label': "Graph Fifteen", 'value': 14}, {'label': "Graph Sixteen", 'value': 15}, {'label': "Graph Seventeen", 'value': 16}, {'label': "Graph Eighteen", 'value': 17}, {'label': "Graph Nineteen", 'value': 18}, {'label': "Graph Twenty", 'value': 19}, {'label': "Graph Twenty-One", 'value': 20}, {'label': "Graph Twenty-Two", 'value': 21}, {'label': "Graph Twenty-Three", 'value': 22}, {'label': "Graph Twenty-Four", 'value': 23}, {'label': "Graph Twenty-Five", 'value': 24}, {'label': "Graph Twenty-Six", 'value': 25}, {'label': "Graph Twenty-Seven", 'value': 26}, {'label': "Graph Twenty-Eight", 'value': 27}, {'label': "Graph Twenty-Nine", 'value': 28}, {'label': "Graph Thirty", 'value': 29}, {'label': "Graph Thirty-One", 'value': 30}, {'label': "Graph Thirty-Two", 'value': 31}, {'label': "Graph Thirty-Three", 'value': 32}, {'label': "Graph Thirty-Four", 'value': 33}, {'label': "Graph Thirty-Five", 'value': 34}, {'label': "Graph Thirty-Six", 'value': 35}, {'label': "Graph Thirty-Seven", 'value': 36}, {'label': "Graph Thirty-Eight", 'value': 37}], "Please select the Model for Graph Classification", "Please select the Model for Graph Classification", [{"label": "Model One. GCN+GAP", "value": "GCN_plus_GAP"}, {"label": "Model Two. DGCNN", "value": "DGCNN"}, {"label": "Model Three. DIFFPOOL", "value": "DIFFPOOL"}, {"label": "Model Four. GIN", "value": "GIN"}], "Please select the Method for Graph Classification", "Please select the Method for Graph Classification", [{"label": "Method One. SA", "value": "SA"}, {"label": "Method Two. GuidedBP", "value": "GuidedBP"}, {"label": "Method Three. CAM", "value": "CAM"}, {"label": "Method Four. Grad-CAM", "value": "Grad-CAM"}, {"label": "Method Five. LRP", "value": "LRP"}, {"label": "Method Six. ExcitationBP", "value": "ExcitationBP"}, {"label": "Method Seven. PGMExplainer", "value": "PGMExplainer"}, {"label": "Method Eight. GNNExplainer", "value": "GNNExplainer"}, {"label": "Method Nine. GraphMask", "value": "GraphMask"}, {"label": "Method Ten. PGExplainer", "value": "PGExplainer"}, {"label": "Method Eleven. SubgraphX", "value": "SubgraphX"}, {"label": "Method Twelve. CF\u00b2", "value": "CF\u00b2"}]
    elif Task_Dropdown == "NC":
        return "Please select the Node", "Please select the Node", ["Node One", "Node Two"], "Please select the Model for Node Classification", "Please select the Model for Node Classification", ["Model One. NC", "Model Two. NC"], "Please select the Method for Node Classification", "Please select the Method for Node Classification", ["Method One. NC", "Method Two. NC"] #"Please select the Graph", "Please select the Graph", [{'label': "Graph One", 'value': 0}, {'label': "Graph Two", 'value': 1}, {'label': "Graph Three", 'value': 2}, {'label': "Graph Four", 'value': 3}, {'label': "Graph Five", 'value': 4}, {'label': "Graph Six", 'value': 5}, {'label': "Graph Seven", 'value': 6}, {'label': "Graph Eight", 'value': 7}, {'label': "Graph Nine", 'value': 8}, {'label': "Graph Ten", 'value': 9}, {'label': "Graph Eleven", 'value': 10}, {'label': "Graph Twelve", 'value': 11}, {'label': "Graph Thirteen", 'value': 12}, {'label': "Graph Fourteen", 'value': 13}, {'label': "Graph Fifteen", 'value': 14}, {'label': "Graph Sixteen", 'value': 15}, {'label': "Graph Seventeen", 'value': 16}, {'label': "Graph Eighteen", 'value': 17}, {'label': "Graph Nineteen", 'value': 18}, {'label': "Graph Twenty", 'value': 19}, {'label': "Graph Twenty-One", 'value': 20}, {'label': "Graph Twenty-Two", 'value': 21}, {'label': "Graph Twenty-Three", 'value': 22}, {'label': "Graph Twenty-Four", 'value': 23}, {'label': "Graph Twenty-Five", 'value': 24}, {'label': "Graph Twenty-Six", 'value': 25}, {'label': "Graph Twenty-Seven", 'value': 26}, {'label': "Graph Twenty-Eight", 'value': 27}, {'label': "Graph Twenty-Nine", 'value': 28}, {'label': "Graph Thirty", 'value': 29}, {'label': "Graph Thirty-One", 'value': 30}, {'label': "Graph Thirty-Two", 'value': 31}, {'label': "Graph Thirty-Three", 'value': 32}, {'label': "Graph Thirty-Four", 'value': 33}, {'label': "Graph Thirty-Five", 'value': 34}, {'label': "Graph Thirty-Six", 'value': 35}, {'label': "Graph Thirty-Seven", 'value': 36}, {'label': "Graph Thirty-Eight", 'value': 37}], "Please select the Model for Graph Classification", "Please select the Model for Graph Classification", [{"label": "Model One. GCN+GAP", "value": "GCN_plus_GAP"}, {"label": "Model Two. DGCNN", "value": "DGCNN"}, {"label": "Model Three. DIFFPOOL", "value": "DIFFPOOL"}, {"label": "Model Four. GIN", "value": "GIN"}], "Please select the Method for Graph Classification", "Please select the Method for Graph Classification", [{"label": "Method One. SA", "value": "SA"}, {"label": "Method Two. GuidedBP", "value": "GuidedBP"}, {"label": "Method Three. CAM", "value": "CAM"}, {"label": "Method Four. Grad-CAM", "value": "Grad-CAM"}, {"label": "Method Five. LRP", "value": "LRP"}, {"label": "Method Six. ExcitationBP", "value": "ExcitationBP"}, {"label": "Method Seven. RelEx", "value": "RelEx"}, {"label": "Method Eight. PGMExplainer", "value": "PGMExplainer"}, {"label": "Method Nine. GraphLime", "value": "GraphLime"}, {"label": "Method Ten. GNNExplainer", "value": "GNNExplainer"}, {"label": "Method Eleven. GraphMask", "value": "GraphMask"}, {"label": "Method Twelve. PGExplainer", "value": "PGExplainer"}, {"label": "Method Thirteen. SubgraphX", "value": "SubgraphX"}, {"label": "Method Fourteen. DnX", "value": "DnX"}, {"label": "Method Fifteen. CF\u00b2", "value": "CF\u00b2"}]
    else:
        return "", "Please Select The Task at First...", [], "", "Please Select The Task at First...", [], "", "Please Select The Task at First...",[]

    #fig = go.Figure([go.Scatter(x=df['date'], y=df['{}'.format(Task_Dropdown)], line=dict(color='firebrick', width=4))])

    #fig.update_layout(title='Stock prices over time', xaxis_title='Dates', yaxis_title='Prices')

    #return fig

#########################################################

def My_DataStructure(Graph, fig_width, fig_height, filling_color, edge_highlight):
    ###                     {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
    my_nodes_name_index = []
    for nods in Graph.x:
        my_nodes_name_index.append(nods.detach().tolist().index(max(nods)))


    my_edges = []
    for nods in Graph.edge_attr:
        my_edges.append(nods.detach().tolist().index(max(nods)))
        
    #color_dict = {0: "Red", 1: "Blue", 2: "Yellow", 3: "Green", 4: "Gray", 5: "Purple", 6: "Pink"}
    my_nodes_dict = {}
    for i in range(len(my_nodes_name_index)):
        my_nodes_dict[str(i)] = dict(color=str(my_nodes_name_index[i]), pos='')

    bond_dict = {0: "aromatic", 1: "single", 2: "double", 3: "triple"}
    # list to define edges

    edges = []

    for i in range(len(Graph.edge_index[0])):
        edges.append((str(int(Graph.edge_index[0][i].detach())), str(int(Graph.edge_index[1][i].detach())),
                      my_edges[i]))

    return Graphize(Graph, my_nodes_dict, my_nodes_name_index, edges, fig_width, fig_height, filling_color, edge_highlight)


#print("my nodes: ", len(my_nodes_dict), my_nodes_dict)
#print('my edges: ', len(edges), edges)
def Graphize(Graph, my_nodes_dict, my_nodes_name_index, edges, fig_width, fig_height, filling_color, edge_highlight):
    #print(my_nodes_dict)
    original_graph = nx.Graph()
    # add nodes and edges
    original_graph.add_nodes_from([(n, {"type": atom_index}) for n, atom_index in zip(my_nodes_dict, my_nodes_name_index)])
    #print('nodes: ', len(original_graph.nodes), original_graph.nodes)
    original_graph.add_edges_from((u, v, {"type": label}) for u, v, label in edges)
    #print('edges: ', len(original_graph.edges), original_graph.edges)
    # for i, node in enumerate(original_graph.nodes):
    # print(node, original_graph.nodes[str(i)], original_graph.nodes[str(i)]['type'])

    posit = nx.spring_layout(original_graph, dim=3)

    return Update_Graph(Graph, original_graph, my_nodes_dict, posit, my_nodes_name_index, fig_width, fig_height, filling_color, edge_highlight)


def Update_Graph(Graph, original_graph, my_nodes_dict, posit, my_nodes_name_index, fig_width, fig_height, filling_color, edge_highlight):
    for i in range(len(original_graph.nodes)):
        #print(original_graph.nodes[str(i)])
        original_graph.nodes[str(i)].update(pos=posit[str(i)], color='')
        #print(original_graph.nodes[str(i)])
    atom_size = 60
    traces_for_edges, edge_marker_trace = Edges_Interactive_Part(original_graph, atom_size, edge_highlight)
    node_trace = Nodes_Interactive_Part(original_graph)

    node_trace, edge_marker_trace, original_graph_annotations = Adjacency(original_graph, my_nodes_name_index, node_trace,
                                                                          edge_marker_trace, traces_for_edges, filling_color,
                                                                          atom_size, edge_highlight)

    fig = Figure_Drawing(Graph, edge_marker_trace, node_trace, traces_for_edges, fig_width, fig_height, original_graph_annotations)
    return fig


def Edges_Interactive_Part(original_graph, atom_size, edge_highlight):
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


        dotSizeConversion = .16 / atom_size                                             # length units per node size
        convertedDotDiameter = atom_size * dotSizeConversion
        lengthFracReduction = convertedDotDiameter / length
        lengthFrac = 1 - lengthFracReduction

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
    edge_importance = []
    for i, edge in enumerate(original_graph.edges.values()):
        edge_colors.append(bond_color_dict[edge['type']])
        #if edge_highlight != None:
        #    edge_importance.append(edge['importance'])

    #colors = ['red'] * len(original_graph.edges)
    #print("len(edge_highlight.keys()): ", len(edge_highlight.keys()), "len(original_graph.edges): ", len(original_graph.edges))

    link_size = [5] * len(original_graph.edges)
    marker_size = [10] * len(original_graph.edges)
    link_type = ['solid'] * len(original_graph.edges)
    if (edge_highlight != False) and (edge_highlight != "Method is Not Selected Yet"):
        print("edge_highlight.keys(): ", edge_highlight)
        for i in range(len(edge_highlight.keys())):
            if edge_highlight[i]:
                link_size[i] = 10
                marker_size[i] = 10
                link_type[i] = 'longdash'
    print("link_size: ", len(link_size))

    #for i in range(len(edge_highlight.keys())):
    #    if edge_highlight[i]:
    #        link_size[i] = 10
    traces_for_edges = []



    for i in range(len(original_graph.edges)):
        traces_for_edges.append(go.Scatter3d(x=test_edge_x[i],
                                             y=test_edge_y[i],
                                             z=test_edge_z[i],
                                             hoverinfo='none',
                                             line=dict(color=edge_colors[i], width=link_size[i], dash=link_type[i]),
                                             mode='lines'
                                             )
                                )



    edge_color_names = ["Aromatic", "Single", "Double", "Triple"]
    edge_color_vals = list(range(len(edge_color_names)))
    edge_num_colors = len(edge_color_vals)


    edge_marker_colorscale = [[0, 'rgb(255, 0, 0)'], [0.25, 'rgb(255, 0, 0)'],
                              [0.25, 'rgb(0, 255, 0)'], [0.5, 'rgb(0, 255, 0)'],
                              [0.5, 'rgb(210,105,30)'], [0.75, 'rgb(210,105,30)'],
                              [0.75, 'rgb(255, 255, 0)'], [1.0, 'rgb(255, 255, 0)']
                              ]



    edge_marker_trace = go.Scatter3d(x=edge_x_marker,
                                     y=edge_y_marker,
                                     z=edge_z_marker,
                                     mode='markers',
                                     hoverinfo='text',
                                     #marker_size=5,
                                     marker=dict(showscale=True, size=marker_size, color=[], colorscale=edge_marker_colorscale,
                                                 reversescale=False,
                                                 cmin=-0.5,
                                                 cmax=3.5,
                                                 colorbar=dict(thickness=15, tickvals=edge_color_vals,
                                                               ticktext=edge_color_names,
                                                               orientation='v',
                                                               title='Bonds',
                                                               xanchor='left',
                                                               titleside='right',
                                                               title_font_family="Times New Roman"),
                                                 line_width=0)
                                     )
    return traces_for_edges, edge_marker_trace



def Nodes_Interactive_Part(original_graph):
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
                                          sizemode='diameter',
                                          size=[],
                                          symbol=[],
                                          colorbar=dict(thickness=10, tickvals=node_color_vals,
                                                        ticktext=node_names,
                                                        orientation='h',
                                                        title='Atoms',
                                                        xanchor='center',
                                                        titleside='top',
                                                        title_font_family="Times New Roman",

                                                        ),
                                          #line=dict(color=["black"], width=[8]),
                                          #line_width=0,
                                          )
                              )


    #  node_color_dict = {0: 'rgb(255, 0, 0)', 1: 'rgb(191, 62, 255)', 2: 'rgb(191, 62, 255)', 3: 'rgb(255, 127, 0)', 4: 'rgb(0, 255, 255)', 5: 'rgb(255, 28, 174)', 6: 'rgb(255, 255, 0)'}
    #node_colors = []
    #print(original_graph.nodes)
    #for i, node in enumerate(original_graph.nodes):
    #    node_colors.append(node_color_dict[edge['type']])


    return node_trace

def Adjacency(original_graph, my_nodes_name_index, node_trace, edge_marker_trace, traces_for_edges, filling_color, atom_size, edge_highlight):
    node_adjacencies = []
    node_text = []
    node_names = {0: 'Carbon', 1: 'Nitrogen', 2: 'Oxygen', 3: 'Fluorine', 4: 'Indium', 5: 'Chlorine', 6: 'Bromine'}
    for node, adjacencies in enumerate(original_graph.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(node_names[my_nodes_name_index[node]] + ' <br># of connections: ' + str(len(adjacencies[1])))


    edge_text = []
    edge_type = []
    bond_dict = {0: "Aromatic", 1: "Single", 2: "Double", 3: "Triple"}

    for i, edge in enumerate(original_graph.edges.values()):
        edge_type.append(edge['type'])
        edge_text.append(bond_dict[edge['type']])


    atoms_size = []
    if (type(filling_color) == dict):

        symbol_types_for_nodes = []
        for i in range(len(filling_color.keys())):
            for j in range(len(filling_color[i].keys())):
                if (filling_color[i][j] == True) or (filling_color[i][j] == [True]):
                    symbol_types_for_nodes.append('circle')
                    atoms_size.append(atom_size)
                else:
                    symbol_types_for_nodes.append('circle-open')
                    atoms_size.append(atom_size)
        #print("filling color: ",  len(filling_color[0]), filling_color)
    else:
        symbol_types_for_nodes = ['circle-open'] * (len(original_graph.nodes))#-1)
        atoms_size = [atom_size] * (len(original_graph.nodes))

    #symbol_types_for_nodes.append('circle')
    #print("atoms_size: ", atoms_size)
    node_trace.marker.color = my_nodes_name_index
    node_trace.marker.size = atoms_size
    #node_trace.marker.line.color = my_nodes_name_index
    node_trace.text = node_text
    node_trace.marker.symbol = symbol_types_for_nodes

    #edge_trace.marker.color = edge_type
    #print(len(edge_type), edge_type)
    #edge_trace.line.color = edge_type
    edge_marker_trace.marker.color = edge_type
    edge_marker_trace.text = edge_text


    ##################################################################     Annotations     #############################
    nick_node_names = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}

    annotations_original_graph = []
    for i in range(len(original_graph.nodes)):
        posit = original_graph.nodes[str(i)]['pos']
        #print("posit: ", posit)
        annotations_original_graph.append(dict(x=posit[0], y=posit[1], z=posit[2],
                                               text=nick_node_names[original_graph.nodes[str(i)]['type']], xanchor='left',
                                               xshift=-12, yshift=+1, font=dict(color='black', size=30),
                                               showarrow=False, arrowhead=1, ax=0, ay=0
                                               )
                                          )



    return node_trace, edge_marker_trace, annotations_original_graph

def Figure_Drawing(Graph, edge_text_trace, node_trace, traces, fig_width, fig_height, original_graph_annotations):
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )#title=dict(text="<b>Figure 2</b>  " + icon, font_family='Arial', yanchor='top', xanchor='left', xref='paper', x=0.01)
    Data = [node_trace, edge_text_trace]
    Data.extend(traces)
    #print("up to figure fine")
    fig = go.Figure(data=Data,#[edge_trace, node_trace, edge_text_trace],
                    layout=go.Layout(plot_bgcolor='white',
                                     width=fig_width, height=fig_height,
                                     #title='<br>Molecule',
                                     title=dict(text="<b>Molecule</b> Label: " + str(Graph.y.detach().tolist()[0]),
                                                y=0.96, xanchor='left', yanchor='top'),
                                     titlefont_size=16,
                                     showlegend=False,
                                     hovermode='closest',
                                     margin=dict(b=20, l=5, r=5, t=40),
                                     scene=dict(annotations=original_graph_annotations, xaxis=dict(axis),
                                                yaxis=dict(axis), zaxis=dict(axis)),
                                     #annotations=[dict(showarrow=False,
                                     #                  xref="paper", yref="paper",
                                     #                  x=0, y=0, xanchor='right', yanchor='top')]
                                    )
                    )
    camera = dict(
        eye=dict(x=1, y=2, z=0.1)
    )

    fig.update_layout(showlegend=False, scene_camera=camera, scene_aspectmode='cube', title_font_family="Times New Roman")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    #fig.show()
    return fig

def graphize_for_isomers(Graph):
    my_isomer_nodes_name_index = []
    for nods in Graph.x:
        my_isomer_nodes_name_index.append(nods.detach().tolist().index(max(nods)))

    my_edges = []
    for nods in Graph.edge_attr:
        my_edges.append(nods.detach().tolist().index(max(nods)))

    my_nodes_dict = {}
    for i in range(len(my_isomer_nodes_name_index)):
        my_nodes_dict[str(i)] = dict(color=str(my_isomer_nodes_name_index[i]), pos='')

    edges = []
    for i in range(len(Graph.edge_index[0])):
        edges.append((str(int(Graph.edge_index[0][i].detach())), str(int(Graph.edge_index[1][i].detach())),
                      my_edges[i]))
    original_graph = nx.Graph()
    original_graph.add_nodes_from(
        [(n, {"type": atom_index}) for n, atom_index in zip(my_nodes_dict, my_isomer_nodes_name_index)])
    original_graph.add_edges_from((u, v, {"type": label}) for u, v, label in edges)


    return original_graph, my_isomer_nodes_name_index

def find_isomers(graph_index, your_dataset):
    target_graph, my_isomer_nodes_name_index = graphize_for_isomers(your_dataset[graph_index])
    print("graphize isomers done")
    #print("target graph: ", target_graph.nodes)


    #your_dataset[graph_index]
    isomers_of_the_target_graph = []
    for i, graph in enumerate(your_dataset):
        new_graph, my_isomer_nodes_name_index = graphize_for_isomers(graph)
        if nx.vf2pp_is_isomorphic(target_graph, new_graph, node_label=None) and i != graph_index:
            isomers_of_the_target_graph.append(1)
        else:
            isomers_of_the_target_graph.append(0)
    return isomers_of_the_target_graph




def isomer_Edges_Interactive_Part(original_graph):
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


    #edge_color_names = ["Aromatic", "Single", "Double", "Triple"]
    edge_color_names = ["Ar", "S", "D", "Tr"]
    edge_color_vals = list(range(len(edge_color_names)))
    edge_num_colors = len(edge_color_vals)


    edge_marker_colorscale = [[0, 'rgb(255, 0, 0)'], [0.25, 'rgb(255, 0, 0)'],
                              [0.25, 'rgb(0, 255, 0)'], [0.5, 'rgb(0, 255, 0)'],
                              [0.5, 'rgb(210,105,30)'], [0.75, 'rgb(210,105,30)'],
                              [0.75, 'rgb(255, 255, 0)'], [1.0, 'rgb(255, 255, 0)']
                              ]



    edge_marker_trace = go.Scatter3d(x=edge_x_marker,
                                     y=edge_y_marker,
                                     z=edge_z_marker,
                                     mode='markers',
                                     hoverinfo='text',
                                     #marker_size=5,
                                     marker=dict(showscale=True, size=5, color=[], colorscale=edge_marker_colorscale,
                                                 reversescale=False,
                                                 cmin=-0.5,
                                                 cmax=3.5,
                                                 colorbar=dict(thickness=5, tickvals=edge_color_vals,
                                                               ticktext=edge_color_names,
                                                               tickfont=dict(size=8),
                                                               orientation='v',
                                                               title='Bonds',
                                                               xanchor='left',
                                                               titleside='right',
                                                               title_font_family="Times New Roman"),
                                                 line_width=0)
                                     )
    return traces_for_edges, edge_marker_trace




def isomer_Nodes_Interactive_Part(original_graph):
    node_x = []
    node_y = []
    node_z = []
    for node in original_graph.nodes():
        x, y, z = original_graph.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)


    # node_names = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
    #node_names = ['Carbon', 'Nitrogen', 'Oxygen', 'Fluorine', 'Indium', 'Chlorine', 'Bromine']
    node_names = ['C', 'N', 'O', 'F', 'In', 'Cl', 'Br']
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
                                          size=14,
                                          sizemode='diameter',
                                          symbol=[],
                                          colorbar=dict(thickness=5, tickvals=node_color_vals,
                                                        ticktext=node_names,
                                                        tickfont=dict(size=8),
                                                        orientation='h',
                                                        title='Atoms',
                                                        xanchor='center',
                                                        titleside='top',
                                                        title_font_family="Times New Roman"
                                                        ),
                                          #line=dict(color=["black"], width=[8]),
                                          line_width=0,
                                          )
                              )


    #  node_color_dict = {0: 'rgb(255, 0, 0)', 1: 'rgb(191, 62, 255)', 2: 'rgb(191, 62, 255)', 3: 'rgb(255, 127, 0)', 4: 'rgb(0, 255, 255)', 5: 'rgb(255, 28, 174)', 6: 'rgb(255, 255, 0)'}
    #node_colors = []
    #print(original_graph.nodes)
    #for i, node in enumerate(original_graph.nodes):
    #    node_colors.append(node_color_dict[edge['type']])


    return node_trace


def isomers_Adjacency(original_graph, my_nodes_name_index, node_trace, edge_marker_trace, traces_for_edges):
    node_adjacencies = []
    node_text = []
    node_names = {0: 'Carbon', 1: 'Nitrogen', 2: 'Oxygen', 3: 'Fluorine', 4: 'Indium', 5: 'Chlorine', 6: 'Bromine'}
    for node, adjacencies in enumerate(original_graph.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(node_names[my_nodes_name_index[node]] + ' <br># of connections: ' + str(len(adjacencies[1])))


    edge_text = []
    edge_type = []
    bond_dict = {0: "Aromatic", 1: "Single", 2: "Double", 3: "Triple"}

    for i, edge in enumerate(original_graph.edges.values()):
        edge_type.append(edge['type'])
        edge_text.append(bond_dict[edge['type']])



    #print(original_graph.nodes)

    symbol_types_for_nodes = ['circle-open'] * (len(original_graph.nodes))

    #print("filling color: ",  len(filling_color[0]), filling_color)
    #symbol_types_for_nodes.append('circle')
    node_trace.marker.color = my_nodes_name_index
    #node_trace.marker.line.color = my_nodes_name_index
    node_trace.text = node_text
    node_trace.marker.symbol = symbol_types_for_nodes

    #edge_trace.marker.color = edge_type
    #print(len(edge_type), edge_type)
    #edge_trace.line.color = edge_type
    edge_marker_trace.marker.color = edge_type
    edge_marker_trace.text = edge_text
    #for i, edge in enumerate(original_graph.edges.values()):
    #    print(i, "  ", edge['type'])


    ##################################################################     Annotations     #############################
    nick_node_names = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}

    annotations_original_graph = []
    for i in range(len(original_graph.nodes)):
        posit = original_graph.nodes[str(i)]['pos']
        annotations_original_graph.append(dict(x=posit[0], y=posit[1], z=posit[2],
                                               text=nick_node_names[original_graph.nodes[str(i)]['type']], xanchor='left',
                                               xshift=-8, yshift=+1, font=dict(color='black', size=17),
                                               showarrow=False, arrowhead=1, ax=0, ay=0
                                               )
                                          )



    return node_trace, edge_marker_trace, annotations_original_graph




def isomers_Figure_Drawing(indexes_for_isomers, Graph, edge_text_trace, node_trace, traces, isomer_fig_width, isomer_fig_height, original_graph_annotations):
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )#title=dict(text="<b>Figure 2</b>  " + icon, font_family='Arial', yanchor='top', xanchor='left', xref='paper', x=0.01)
    fake_job = Graph.y.detach().tolist()[0]
    if fake_job == 1:
        fake_job = 0
    elif fake_job == 0:
        fake_job = 1
    Data = [node_trace, edge_text_trace]
    Data.extend(traces)
    print("extended done")
    #print("up to figure fine")
    fig = go.Figure(data=Data,#[edge_trace, node_trace, edge_text_trace],
                    layout=go.Layout(plot_bgcolor='white',
                                     width=isomer_fig_width, height=isomer_fig_height,
                                     #title='<b>' + str(len(indexes_for_isomers)) + ' iSomers',
                                     title=dict(text="<b>Molecule</b> Label: " + str(fake_job) + ' <br>has <b>' + str(len(indexes_for_isomers)) + ' iSomer/s', y=0.96, xanchor='left', yanchor='top'),
                                     titlefont_size=12,
                                     showlegend=False,
                                     hovermode='closest',
                                     margin=dict(b=5, l=5, r=0, t=5),
                                     scene=dict(annotations=original_graph_annotations, xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis)),
                                     annotations=[dict(showarrow=False,
                                                       xref="paper", yref="paper",
                                                       x=0, y=0, xanchor='right', yanchor='top')]
                                    )
                    )
    fig.update_layout(showlegend=False, title_font_family="Times New Roman")

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    #fig.show()
    return fig


def display_my_isomers(indexes_for_isomers, dataset, selected_index_for_isomer, isomer_fig_width, isomer_fig_height):
    #if selected_index_for_isomer >= 0:
    graph_form_of_isomers, my_isomer_nodes_name_index = graphize_for_isomers(dataset[indexes_for_isomers[selected_index_for_isomer]])

    isomer_posit = nx.spring_layout(graph_form_of_isomers, dim=3)

    for i in range(len(graph_form_of_isomers.nodes)):
        graph_form_of_isomers.nodes[str(i)].update(pos=isomer_posit[str(i)], color='')


    isomers_traces_for_edges, isomers_edge_marker_trace = isomer_Edges_Interactive_Part(graph_form_of_isomers)
    isomers_node_trace = isomer_Nodes_Interactive_Part(graph_form_of_isomers)


    node_trace, edge_marker_trace, original_graph_annotations = isomers_Adjacency(graph_form_of_isomers,
                                                                                  my_isomer_nodes_name_index,
                                                                                  isomers_node_trace, isomers_edge_marker_trace,
                                                                                  isomers_traces_for_edges)

    isomer_fig = isomers_Figure_Drawing(indexes_for_isomers, dataset[indexes_for_isomers[selected_index_for_isomer]], edge_marker_trace, node_trace, isomers_traces_for_edges,
                                        isomer_fig_width, isomer_fig_height, original_graph_annotations)

    return isomer_fig



def isomer_empty_fig(iso_fig_width, iso_fig_height):
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )
    fig = go.Figure(data=[],
                    layout=go.Layout(plot_bgcolor='white',
                                     width=iso_fig_width, height=iso_fig_height,
                                     title='<b>Menu to Select...</b>',
                                     titlefont_size=10,
                                     showlegend=False,
                                     hovermode='closest',
                                     margin=dict(b=20, l=5, r=5, t=40),
                                     scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis)),
                                     annotations=[dict(showarrow=False,
                                                       xref="paper", yref="paper",
                                                       x=0, y=0, xanchor='right', yanchor='top')]
                                     )
                    )
    fig.update_layout(title_font_family="Times New Roman")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig

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
    subgraph_edge_trace = subgraph_Edges_Interactive_Part(subgraph_original)
    subgraph_node_trace = subgraph_Nodes_Interactive_Part(subgraph_original)
    subgraph_node_trace, subgraph_edge_trace, subgraph_graph_annotations = subgraph_Adjacency(subgraph_original, subgraph_my_nodes_name_index, subgraph_node_trace, subgraph_edge_trace)

    subgraph_fig = subgraph_Figure_Drawing(subgraph_node_trace, subgraph_edge_trace, sub_fig_width, sub_fig_height, subgraph_graph_annotations)
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

    #bond_color_dict = {0: 'rgb(255, 0, 0)', 1: 'rgb(0, 255, 0)', 2: 'rgb(210,105,30)', 3: 'rgb(255, 255, 0)'}
    #edge_colors = []
    #for i, edge in enumerate(original_graph.edges.values()):
    #    edge_colors.append(bond_color_dict[edge['type']])

    #colors = ['red'] * len(original_graph.edges)
    link_size = [5] * len(original_graph.edges)
    traces_for_edges = []



    for i in range(len(original_graph.edges)):
        traces_for_edges.append(go.Scatter3d(x=test_edge_x[i],
                                             y=test_edge_y[i],
                                             z=test_edge_z[i],
                                             hoverinfo='none',
                                             line=dict(color='black', width=link_size[i]),
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
    #node_names = ['Carbon', 'Nitrogen', 'Oxygen', 'Fluorine', 'Indium', 'Chlorine', 'Bromine']
    node_names = ['C', 'N', 'O', 'Fl', 'In', 'Cl', 'Br']
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
                                          size=14,
                                          sizemode='diameter',
                                          symbol=[],
                                          colorbar=dict(thickness=5, tickvals=node_color_vals,
                                                        ticktext=node_names,
                                                        tickfont=dict(size=8),
                                                        orientation='h',
                                                        title='Atoms',
                                                        xanchor='center',
                                                        titleside='top',
                                                        title_font_family="Times New Roman"
                                                        ),
                                          #line=dict(color=["black"], width=[8]),
                                          #line_width=0,
                                          )
                              )
    return node_trace










def subgraph_Adjacency(original_graph, my_nodes_name_index, node_trace, subgraph_edge_trace):
    node_adjacencies = []
    node_text = []
    #node_names = {0: 'Carbon', 1: 'Nitrogen', 2: 'Oxygen', 3: 'Fluorine', 4: 'Indium', 5: 'Chlorine', 6: 'Bromine'}
    node_names = {0: 'C', 1: 'N', 2: 'O', 3: 'Fl', 4: 'In', 5: 'Cl', 6: 'Br'}
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
                                               xshift=-8, yshift=+1, font=dict(color='black', size=17),
                                               showarrow=False, arrowhead=1, ax=0, ay=0
                                               )
                                          )



    return node_trace, subgraph_edge_trace, annotations_original_graph





def subgraph_Figure_Drawing(node_trace, subgraph_edge_trace, sub_fig_width, sub_fig_height, original_graph_annotations):
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )

    Data = [node_trace]
    Data.extend(subgraph_edge_trace)


    fig = go.Figure(data=Data,
                    layout=go.Layout(plot_bgcolor='white',
                                     width=sub_fig_width, height=sub_fig_height,
                                     title=dict(text="<b>Subgraph</b> Label: ", y=0.96, xanchor='left', yanchor='top',),
                                     titlefont_size=10,
                                     showlegend=False,
                                     hovermode='closest',
                                     margin=dict(b=5, l=5, r=0, t=5),
                                     scene=dict(annotations=original_graph_annotations, xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis)),
                                     #annotations=[dict(showarrow=False,
                                     #                  xref="paper", yref="paper",
                                     #                  x=0, y=0, xanchor='right', yanchor='top')]
                                    )
                    )
    fig.update_layout(showlegend=False, title_font_family="Times New Roman")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig

def subgraph_empty_fig(subgraph_fig_width, subgraph_fig_height):
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )
    fig = go.Figure(data=[],
                    layout=go.Layout(plot_bgcolor='white',
                                     width=subgraph_fig_width, height=subgraph_fig_height,
                                     title='<b>Method to Select...</b>',
                                     titlefont_size=10,
                                     showlegend=False,
                                     hovermode='closest',
                                     margin=dict(b=20, l=5, r=5, t=40),
                                     scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis)),
                                     annotations=[dict(showarrow=False,
                                                       xref="paper", yref="paper",
                                                       x=0, y=0, xanchor='right', yanchor='top')]
                                     )
                    )
    fig.update_layout(title_font_family="Times New Roman")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig

@app.callback([#Output(component_id='iSomer_Dropdown', component_property='value'),
               Output(component_id='iSomer_Dropdown', component_property='placeholder'),
               Output(component_id='iSomer_Dropdown', component_property='options')],
              [Input(component_id='my_display_toggle', component_property='on'),
               Input(component_id='Task_Dropdown', component_property='value'),
               Input(component_id='Index_Dropdown', component_property='value'),
               Input(component_id='iSomer_Dropdown', component_property='value')])
def update_isomers_deropdown(on, selected_task, selected_index, selected_isomer):
    #print('=============this is the selected isomer', selected_isomer)

    if on and (selected_task == "GC") and (selected_index != None) and (
            selected_index != 'Please Select The Task at First...') and (selected_index != "Please select the Graph"):
        isomers_of_the_selected_graph = find_isomers(test_indexes_read[selected_index], Mutag_Dataset)
        count_for_isomers_of_the_selected_graph = sum(isomers_of_the_selected_graph)
        isomer_options = []
        for i in range(count_for_isomers_of_the_selected_graph):
            isomer_options.append({'label': "iSomer "+str(i+1), 'value': i})
        if len(isomer_options) > 0:
            return "iSomers", isomer_options
        else:
            return "No iSomer", []
    elif on and (selected_task == "GC") and (selected_index != None) and (
            selected_index != 'Please Select The Task at First...') and (selected_index != "Please select the Graph") and \
            (selected_isomer == None or selected_isomer == "" or selected_isomer == "iSomers"
             or selected_isomer == "Please Select The iSomer"):
        #print('Im here ========================')
        return "iSomers", []
    else:
        #print('Im here 2========================')
        return "iSomers", []

@app.callback([#Output(component_id='iSomer_Dropdown', component_property='value'),
               Output(component_id='Subgraph_Dropdown', component_property='placeholder'),
               Output(component_id='Subgraph_Dropdown', component_property='options')],
              [Input(component_id='my_display_toggle', component_property='on'),
               Input(component_id='Task_Dropdown', component_property='value'),
               Input(component_id='Method_Dropdown', component_property='value')])
def update_subgraph_dropdown(on, selected_task, selected_method):
    if on and (selected_task == "GC") and (selected_method != None) and (
            selected_method != 'Please Select The Task at First...') and (
            selected_method != "Please select the Method for Graph Classification"):
        return "Subgraphs", [{'label': "Class One Subgraph One", 'value': 0}, {'label': "Class One Subgraph Two", 'value': 1},
                             {'label': "Class Zero Subgraph One", 'value': 2}, {'label': "Class Zero Subgraph Two", 'value': 3}]
    else:
        return "Please select method", []



@app.callback(Output(component_id='download_your_pretrained_model_weights', component_property='data'),
              Input(component_id='Download_Button', component_property='n_clicks'), prevent_initial_call=True,)
def download_pretrained_model(download_weights_clicks):
    if download_weights_clicks:
        print("did you click me? then take your file...", download_weights_clicks)
        return dcc.send_file(
            "/Users/EY33JW/PycharmProjects/pythonProject2/pretrained_model_weights_to_be_downloaded.pt"
        )


@app.callback([Output(component_id='Visualize_Graph', component_property='children'),
               Output(component_id='Visualize_iSomer_Graph', component_property='children'),
               Output(component_id='Visualize_Subgraph_Graph', component_property='children'),
               Output(component_id='my_display_toggle', component_property='on')],
              [Input(component_id='my_display_toggle', component_property='on'),
               Input(component_id='Task_Dropdown', component_property='value'),
               Input(component_id='Index_Dropdown', component_property='value'),
               Input(component_id='Method_Dropdown', component_property='value'),
               Input(component_id='importance_threshold_my-slider', component_property='value'),
               Input(component_id='iSomer_Dropdown', component_property='value'),
               Input(component_id='Subgraph_Dropdown', component_property='value')])
def display_my_figure(on, task_value, index_value, selected_method, threshold_slider, selected_index_for_isomer, selected_index_for_subgraph):
    print(on, task_value, index_value)
    #subgraph_fig_width = 200
    #subgraph_fig_height = 200
    #selected_subgraph = 0
    #print("the seleceted index for the isomer: ", type(selected_index_for_isomer))
    #selected_index_for_isomer = 0
    #print("========this is the selected subgraph: ", selected_index_for_subgraph)


    ##########################################################         SubGraph         ################################

    if on and (task_value == "GC") and (selected_method != None) and (
            selected_method != 'Please Select The Task at First...') and (
            selected_method != "Please select the Method for Graph Classification") and (selected_index_for_subgraph != None) and (selected_index_for_subgraph != 'Subgraphs') and (selected_index_for_subgraph != "Please select method") and (selected_index_for_subgraph != ""):
        subgraph_fig = subgraph_datastructure(selected_index_for_subgraph, subgraph_fig_width, subgraph_fig_height)
    else:
        #print("came here")
        subgraph_fig = subgraph_empty_fig(subgraph_fig_width, subgraph_fig_height)


    ##########################################################         iSomers          ################################

    if on and (task_value == "GC") and (index_value != None) and (index_value != 'Please Select The Task at First...') and (index_value != "Please select the Graph"):
        isomers_of_the_selected_graph = find_isomers(test_indexes_read[index_value], Mutag_Dataset)
        count_for_isomers_of_the_selected_graph = sum(isomers_of_the_selected_graph)
        #print("isomers of the seleceted graph:", isomers_of_the_selected_graph)
        #print('number of isomers: ', count_for_isomers_of_the_selected_graph)

        indexes_for_isomers = []
        for i in range(len(isomers_of_the_selected_graph)):
            if isomers_of_the_selected_graph[i] == 1:
                indexes_for_isomers.append(i)
        #if len(indexes_for_isomers) != 0:
        #selected_index_for_isomer = 0  # dropdown
        if (selected_index_for_isomer != "Please select iSomer...") and (selected_index_for_isomer != "Please Select The Task and Graph...") and (selected_index_for_isomer != None) and (selected_index_for_isomer != ""):
            #print("this is the seleceted isomer: ", selected_index_for_isomer,)
            isomer_fig = display_my_isomers(indexes_for_isomers, Mutag_Dataset, selected_index_for_isomer, isomer_fig_width, isomer_fig_height)
        else:
            #print("isomer is not selected", selected_index_for_isomer,)
            isomer_fig = isomer_empty_fig(isomer_fig_width, isomer_fig_height)
    else:
        isomer_fig = isomer_empty_fig(isomer_fig_width, isomer_fig_height)


    #########################################################        Main Screen        ################################

    if on and (task_value == "GC") and (index_value != None) and (index_value != 'Please Select The Task at First...')\
            and (index_value != "Please select the Graph"):

        #print("real index of the selected graph: ", test_indexes_read[index_value])
        # print(test_indexes_read)


        #     importance_threshold = 0.5
        Graph = Graphs_Family[index_value]

        if selected_method == "SA":
            new_output = SA_Method_as_Class.SA_GC(task="Graph Classification", method="SA", model_name="GCN_plus_GAP",
                                                  graph=[Graph], importance_threshold=threshold_slider, load_index=200,
                                                  input_dim=len(Graph.x[0]), hid_dim=7, output_dim=2, normalize_coeff=10)
            print(selected_method, new_output.saliency_maps)
            print(selected_method, new_output.importance_dict)
            Saliency_Maps_Continous = new_output.saliency_maps
            Saliency_Maps_Dict = new_output.importance_dict
            filling_color = Saliency_Maps_Dict
            edges_highlight = False

        elif selected_method == "GuidedBP":
            new_output = GuidedBP_Method_as_Class.GuidedBP_GC(task="Graph Classification", method="GuidedBP",
                                                              model_name="GCN_plus_GAP", graph=[Graph],
                                                              importance_threshold=threshold_slider, load_index=200,
                                                              input_dim=len(Graph.x[0]), hid_dim=7, output_dim=2,normalize_coeff=10)
            print(selected_method, selected_method, new_output.saliency_maps)
            print(selected_method, new_output.importance_dict)
            Saliency_Maps_Continous = new_output.saliency_maps
            Saliency_Maps_Dict = new_output.importance_dict
            filling_color = Saliency_Maps_Dict
            edges_highlight = False

        elif selected_method == "CAM":
            new_output = CAM_Method_as_Class.CAM_GC(task="Graph Classification", method="CAM", model_name="GCN_plus_GAP",
                                                    graph=[Graph], importance_threshold=threshold_slider,
                                                    load_index=200, input_dim=len(Graph.x[0]), hid_dim=7, output_dim=2,
                                                    normalize_coeff=10, DataSet_name = "MUTAG")
            print("selected_method: ", selected_method, selected_method, new_output.saliency_maps)
            print(selected_method, new_output.importance_dict)
            Saliency_Maps_Continous = new_output.saliency_maps
            Saliency_Maps_Dict = new_output.importance_dict
            filling_color = Saliency_Maps_Dict
            edges_highlight = False

        elif selected_method == "Grad-CAM":
            new_output = Grad_CAM_Method_as_Class.Grad_CAM_GC(task="Graph Classification", method="Grad-CAM",
                                                              model_name="GCN_plus_GAP", graph=[Graph],
                                                              importance_threshold=threshold_slider, load_index=200,
                                                              input_dim=len(Graph.x[0]), hid_dim=7, output_dim=2,
                                                              normalize_coeff=100, DataSet_name = "MUTAG")
            print(selected_method, selected_method, new_output.saliency_maps)
            print(selected_method, new_output.importance_dict)
            Saliency_Maps_Continous = new_output.saliency_maps
            Saliency_Maps_Dict = new_output.importance_dict
            filling_color = Saliency_Maps_Dict
            edges_highlight = False

        elif selected_method == "LRP":
            new_output = LRP_Method_as_Class.LRP_GC(task="Graph Classification", method="LRP", model_name="GCN_plus_GAP",
                                                    graph=[Graph], importance_threshold=threshold_slider, load_index=200,
                                                    input_dim=len(Graph.x[0]), hid_dim=7, output_dim=2, normalize_coeff=100,
                                                    DataSet_name="MUTAG")
            print(selected_method, selected_method, new_output.saliency_maps)
            print(selected_method, new_output.importance_dict)
            Saliency_Maps_Continous = new_output.saliency_maps
            Saliency_Maps_Dict = new_output.importance_dict
            filling_color = Saliency_Maps_Dict
            edges_highlight = False

        elif selected_method == "ExcitationBP":
            new_output = ExcitationBP_Method_as_Class.ExcitationBP_GC(task="Graph Classification", method="ExcitationBP",
                                                                      model_name="GCN_plus_GAP", graph=[Graph],
                                                                      importance_threshold=threshold_slider,
                                                                      load_index=200,input_dim=len(Graph.x[0]),
                                                                      hid_dim=7, output_dim=2, normalize_coeff=100,
                                                                      DataSet_name="MUTAG")
            print(selected_method, selected_method, new_output.saliency_maps)
            print(selected_method, new_output.importance_dict)
            Saliency_Maps_Continous = new_output.saliency_maps
            Saliency_Maps_Dict = new_output.importance_dict
            filling_color = Saliency_Maps_Dict
            edges_highlight = False

        elif selected_method == "GNNExplainer":
            new_output = GNNExplainer_Method_as_Class.GNNExplainer_GC(task="Graph Classification", method="GNNExplainer",
                                                                      model_name="GCN_plus_GAP", graph=[Graph],
                                                                      importance_threshold=threshold_slider, load_index=200,
                                                                      input_dim=len(Graph.x[0]), hid_dim=7, output_dim=2,
                                                                      normalize_coeff=100, DataSet_name="MUTAG")
            print(selected_method, selected_method, new_output.saliency_maps)
            print(selected_method, new_output.importance_dict)
            Saliency_Maps_Continous = new_output.saliency_maps
            Saliency_Maps_Dict = new_output.importance_dict
            filling_color = Saliency_Maps_Dict
            edges_highlight = new_output.edges_importance_dict

        elif selected_method == "PGExplainer":
            new_output = PGExplainer_Method_as_Class.PGExplainer(Model_Name="GCN_plus_GAP", Explainability_name='PGExplainer',
                                                                 Task_name='Graph Classification', classifier_load_index=200,
                                                                 explainer_save_index=10000, Exp_Epoch=100, Exp_lr=0.001,
                                                                 input_dim=len(Graph.x[0]), hid_dim=7, output_dim=2,
                                                                 importance_threshold=threshold_slider, ExTrain_or_ExTest="test",
                                                                 Exp_Load_index=100, your_dataset=[Graph], target_class="correct",
                                                                 DataSet_name="MUTAG", classifier_save_index=200)



            print(selected_method, selected_method, new_output.saliency_maps)
            print(selected_method, new_output.importance_dict)
            Saliency_Maps_Continous = new_output.saliency_maps
            Saliency_Maps_Dict = new_output.importance_dict
            filling_color = "Method is Not Selected Yet"
            edges_highlight = Saliency_Maps_Dict

        elif selected_method == "GraphMask":
            new_output = GraphMask_Method_as_Class.GraphMask(Model_Name="GCN_plus_GAP", Explainability_name='GraphMask',
                                                             Task_name='Graph Classification', classifier_load_index=200,
                                                             explainer_save_index=50, Exp_Epoch=50, Exp_lr=0.001,
                                                             explainer_hid_dim=7, input_dim=len(Graph.x[0]), hid_dim=7,
                                                             output_dim=2,DataSet_name="MUTAG", importance_threshold=threshold_slider,
                                                             Exp_Load_index=100, ExTrain_or_ExTest="test", your_dataset=[Graph],
                                                             target_class="correct")



            print(selected_method, selected_method, new_output.saliency_maps)
            print(selected_method, new_output.importance_dict)
            Saliency_Maps_Continous = new_output.saliency_maps
            Saliency_Maps_Dict = new_output.importance_dict
            filling_color = "Method is Not Selected Yet"
            edges_highlight = Saliency_Maps_Dict

        elif selected_method == "SubgraphX":
            new_output = SubGraphX_offline_Method_as_Class.SubGraphX_off_the_fly(your_dataset=Graph,
                                                                                 Task_name='Graph Classification',
                                                                                 Model_Name="GCN_plus_GAP", classifier_load_index=200,
                                                                                 input_dim=7, hid_dim=7, output_dim=2,
                                                                                 loading_graph_index=index_value+1,
                                                                                 category='correct', DataSet_name="MUTAG")

            print(selected_method, selected_method, new_output.saliency_maps)
            print(selected_method, new_output.importance_dict)
            Saliency_Maps_Continous = new_output.saliency_maps
            Saliency_Maps_Dict = new_output.importance_dict
            filling_color = Saliency_Maps_Dict
            edges_highlight = False
        elif selected_method == "PGMExplainer":
            new_output = PGMExplainer_Method_as_Class.PGM_Graph_Explainer(Model_Name="GCN_plus_GAP", classifier_load_index=200,
                                                                    input_dim=7, hid_dim=7, output_dim=2,graph=Graph,
                                                                    DataSet_name="MUTAG", perturb_feature_list=[None],
                                                                    perturb_mode="mean", perturb_indicator="abs",
                                                                    importance_threshold=threshold_slider)
            print(selected_method, selected_method, new_output.saliency_maps)
            print(selected_method, new_output.importance_dict)
            Saliency_Maps_Continous = new_output.saliency_maps
            Saliency_Maps_Dict = new_output.importance_dict
            filling_color = Saliency_Maps_Dict
            edges_highlight = False
        elif selected_method == "CF\u00b2":
            new_output = CF2Explainer_Method_as_Class.CF2_Explaination(Model_Name="GCN_plus_GAP", your_dataset=[Graph],
                                                                       explainer_epochs=1000, fix_exp=None,
                                                                       classifier_load_index=200, DataSet_name="MUTAG",
                                                                       input_dim=7, hid_dim=7, output_dim=2,
                                                                       importance_threshold=threshold_slider)
            print(selected_method, selected_method, new_output.saliency_maps)
            print(selected_method, new_output.importance_dict)
            Saliency_Maps_Continous = new_output.saliency_maps
            Saliency_Maps_Dict = new_output.importance_dict
            filling_color = "Method is Not Selected Yet"
            edges_highlight = Saliency_Maps_Dict

        elif (selected_method != "SA") and (selected_method != "GuidedBP") and (selected_method != "CAM") and (selected_method != "Grad-CAM") and (selected_method != "LRP") and (selected_method != "ExcitationBP") and (selected_method != "GNNExplainer"):
            filling_color = "Method is Not Selected Yet"
            edges_highlight = "Method is Not Selected Yet"
            print("filling_color = Method is Not Selected Yet")

        fig = My_DataStructure(Graph, fig_width, fig_height, filling_color, edges_highlight)
        # print(fig)
        dcc.Graph(figure=fig)
        return [dcc.Graph(figure=fig)], [dcc.Graph(figure=isomer_fig)], [dcc.Graph(figure=subgraph_fig)], True

        #Graph = Mutag_Dataset[0]Saliency_Maps_Dict
        #print(Graph.x)
        #fig = My_DataStructure(Graph, fig_width, fig_height)
        #tips = px.data.tips()


    else:
        # isomer_fig = display_my_isomers(indexes_for_isomers, Mutag_Dataset, selected_index_for_isomer)
        isomer_fig = isomer_empty_fig(isomer_fig_width, isomer_fig_height)
        axis = dict(showbackground=False,
                    showline=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    title=''
                    )
        fig = go.Figure(data=[],
                        layout=go.Layout(plot_bgcolor='white',
                                         width=fig_width, height=fig_height,
                                         title='<b>Menu to Select...</b>',
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
        fig.update_layout(title_font_family="Times New Roman")
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        return [dcc.Graph(figure=fig)], [dcc.Graph(figure=isomer_fig)], [dcc.Graph(figure=subgraph_fig)], False




#Graphs_Family = []
#Mutag_Dataset = TUDataset(root='data/TUDataset', name='MUTAG')
#print(Mutag_Dataset[187])
#for i in range(10):
#    Graphs_Family.append(Mutag_Dataset[random.randint(0, len(Mutag_Dataset)-1)])

#Graph = Mutag_Dataset[0]
#fig = My_DataStructure(Graph, fig_width, fig_height)
#print(Graph.x)



####################################################     NETWORK DISPLAY   #############################################
dict1 = \
    {
        '0': '\u2070', '1': '\u00b9', '2': '\u00b2',
        '3': '\u00b3', '4': '\u2074', '5': '\u2075',
        '6': '\u2076', '7': '\u2077', '8': '\u2078',
        '9': '\u2079', '+': '\u207A', '-': '\u207B',
        '=': '\u207C', '(': '\u207D', ')': '\u207E',
        'a': '\u1d43', 'b': '\u1d47', 'c': '\u1D9C',
        'd': '\u1d48', 'e': '\u1d49', 'f': '\u1da0',
        'g': '\u1d4d', 'h': '\u02b0', 'i': '\u2071',
        'j': '\u02b2', 'k': '\u1d4f', 'l': '\u02e1',
        'm': '\u1d50', 'n': '\u207f', 'o': '\u1d52',
        'p': '\u1d56', 'r': '\u02b3', 's': '\u02e2',
        't': '\u1d57', 'u': '\u1d58', 'v': '\u1d5b',
        'w': '\u02b7', 'x': '\u02e3', 'y': '\u02b8',
        'z': '\u1dbb', 'A': '\u1D2c', 'B': '\u1D2E',
        'D': '\u1d30', 'E': '\u1d31', 'G': '\u1d33',
        'H': '\u1d34', 'I': '\u1d35', 'J': '\u1d36',
        'K': '\u1d37', 'L': '\u1d38', 'M': '\u1d39',
        'N': '\u1d3a', 'O': '\u1d3c', 'P': '\u1d3e',
        'R': '\u1d3f', 'T': '\u1d40', 'U': '\u1d41',
        'V': '\u2c7d', 'W': '\u1d42', 'q': '\u146b'
         }


dict2 = \
    {
        '0': '\u2080', '1': '\u2081', '2': '\u2082',
        '3': '\u2083', '4': '\u2084', '5': '\u2085',
        '6': '\u2086', '7': '\u2087', '8': '\u2088',
        '9': '\u2089', '+': '\u208A', '-': '\u208B',
        '=': '\u208C', '(': '\u208D', ')': '\u208E',
        'a': '\u2090', 'e': '\u2091', 'h': '\u2095',
        'i': '\u1d62', 'j': '\u2c7c', 'k': '\u2096',
        'l': '\u2097', 'm': '\u2098', 'n': '\u2099',
        'o': '\u2092', 'p': '\u209A', 'r': '\u1d63',
        's': '\u209B', 't': '\u209C', 'u': '\u1d64',
        'v': '\u1d65', 'x': '\u2093', 'y': '\u1d67'
         }
def supscript(base, x):
    z = '{}'.format(dict1.get(x))
    return base + z

def subscript(base,x):
    z = '{}'.format(dict2.get(x))
    return base + z







def Network_datastructure_part(Scaled_Activations_formatted_list, nonScaled_Activations_formatted_list, frequency_of_atoms, Weights, prediction, net_fig_width, net_fig_height):
    number_of_layers = 2
    number_of_nodes_on_each_layer = [7, 7, 7, 7, 2]

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
    #print(my_nodes_name_index_together)
    #print(my_nodes_name_index_layers)


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
    #print(x_layer_step, y_node_step)
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


    #print("number of position coordinates: ", len(my_nodes_pos_dict), my_nodes_pos_dict)

    edges = []
    for i in range(len(my_nodes_name_index_layers)):
        if i < len(my_nodes_name_index_layers)-1:
            for j in range(len(my_nodes_name_index_layers[i])):
                for k in range(len(my_nodes_name_index_layers[i+1])):
                    edges.append(((str(my_nodes_name_index_layers[i][j])), str(my_nodes_name_index_layers[i+1][k])))
    #print("my edges: ", len(edges), edges)
    #return Network_Graphize(number_of_nodes_on_each_layer, my_nodes_dict_together, my_nodes_name_index_together, edges, fig_width, fig_height)
    return Network_Graphize(Scaled_Activations_formatted_list, nonScaled_Activations_formatted_list, frequency_of_atoms, Weights, prediction, my_nodes_pos_dict, my_nodes_dict_together, my_nodes_name_index_together, edges,
                     net_fig_width, net_fig_height, my_nodes_name_index_layers)

def Network_Graphize(Scaled_Activations_formatted_list, nonScaled_Activations_formatted_list, frequency_of_atoms, Weights, prediction, my_nodes_pos_dict, my_nodes_dict_together, my_nodes_name_index_together, edges, net_fig_width, net_fig_height, my_nodes_name_index_layers):
    network_graph = nx.Graph()


    network_graph.add_nodes_from([n for n in my_nodes_dict_together])
    network_graph.add_edges_from((u, v) for u, v in edges)

    posit = nx.spring_layout(network_graph, dim=2)

    for i in range(len(network_graph.nodes)):
        #print(network_graph.nodes[str(i)])
        network_graph.nodes[str(i)].update(pos=my_nodes_pos_dict[str(i)], color='')
        #print(network_graph.nodes[str(i)])

    edge_trace, edge_text_trace = Network_Edges_Interactive_Part(network_graph)
    #print("Edge Trace works well")
    input_node_trace, hidden_node_trace = Network_Nodes_Interactive_Part(network_graph)
    #print("Node Trace works well")
    input_node_trace, hidden_node_trace, edge_trace, edge_text_trace, annotations = Network_Adjacency(Scaled_Activations_formatted_list, nonScaled_Activations_formatted_list, frequency_of_atoms, Weights, network_graph, input_node_trace, hidden_node_trace, edge_trace, edge_text_trace, my_nodes_name_index_layers)
    # print("Network Adjacency works well")
    #return node_trace, edge_trace,

    # fig = Figure_Drawing(edge_trace, edge_text_trace, node_trace, fig_width, fig_height)
    #return fig
    return Network_Figure_Drawing(edge_trace, edge_text_trace, input_node_trace, hidden_node_trace, net_fig_width, net_fig_height, prediction, annotations)

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
                                   marker=dict(showscale=False, size=1, color='grey', #colorscale='rgb(0, 0, 255)',
                                               reversescale=False, line_width=0)
                                     )
    return edge_trace, edge_marker_trace


def Network_Nodes_Interactive_Part(network_graph):
    #######################################################----Input----################################################
    input_node_x = []
    input_node_y = []

    for i in range(7):
        posit = network_graph.nodes[str(i)]['pos']
        input_node_x.append(posit[0])
        input_node_y.append(posit[1])


    network_node_colorscale = [[0, 'rgb(255, 0, 0)'], [0.14285714, 'rgb(255, 0, 0)'],
                               [0.14285714, 'rgb(0, 255, 0)'], [0.28571429, 'rgb(0, 255, 0)'],
                               [0.28571429, 'rgb(191, 62, 255)'], [0.42857143, 'rgb(191, 62, 255)'],
                               [0.42857143, 'rgb(255, 127, 0)'], [0.57142857, 'rgb(255, 127, 0)'],
                               [0.57142857, 'rgb(0, 255, 255)'], [0.71428571, 'rgb(0, 255, 255)'],
                               [0.71428571, 'rgb(255, 28, 174)'], [0.85714286, 'rgb(255, 28, 174)'],
                               [0.85714286, 'rgb(255, 255, 0)'], [1.0, 'rgb(255, 255, 0)']
                               ]


    input_node_trace = go.Scatter(x=input_node_x,
                                  y=input_node_y,
                                  mode='markers',
                                  # textposition="middle left",
                                  # textfont=dict(size=10),
                                  # text=[],
                                  hoverinfo='text',
                                  marker=dict(showscale=False,
                                              #colorscale='rgb(0, 255, 0)',
                                              colorscale=network_node_colorscale,
                                              reversescale=False,
                                              # cmin=-0.5,
                                              # cmax=6.5,
                                              # color='rgb(255, 0, 0)',
                                              color=[],
                                              size=30,
                                              line=dict(color="#0000ff"),
                                              line_width=2
                                              )
                                  )


    #######################################################----Hidden----###############################################

    hidden_neurons_pos_x = []
    hidden_neurons_pos_y = []

    for i in range(7, len(network_graph.nodes)):
        posit = network_graph.nodes[str(i)]['pos']
        hidden_neurons_pos_x.append(posit[0])
        hidden_neurons_pos_y.append(posit[1])


    #showarrow = [False] * len(network_graph.nodes)
    #annotations = dict(x=hidden_neurons_pos_x, y=hidden_neurons_pos_y, text=network_graph.nodes, showarrow=showarrow)


    hidden_node_trace = go.Scatter(x=hidden_neurons_pos_x,
                                   y=hidden_neurons_pos_y,
                                   mode='markers',
                                   # textposition="middle left",
                                   # textfont=dict(size=10),
                                   # text=[],
                                   hoverinfo='text',
                                   marker=dict(showscale=True,
                                               #colorscale='rgb(0, 255, 0)',
                                               colorscale='Reds',
                                               reversescale=False,
                                               #cmin=-0.5,
                                               #cmax=6.5,
                                               #color='rgb(255, 0, 0)',
                                               color=[],
                                               size=30,
                                               colorbar=dict(thickness=15,
                                                             orientation='v',
                                                             title='Contribution Level',
                                                             title_font_family="Times New Roman",
                                                             xanchor='center',
                                                             titleside='top'),
                                               )
                                   )
    return input_node_trace, hidden_node_trace




def Network_Adjacency(Scaled_Activations_formatted_list, nonScaled_Activations_formatted_list, frequency_of_atoms, Weights, network_graph, input_node_trace, hidden_node_trace, edge_trace, edge_text_trace, my_nodes_name_index_layers):
    input_node_adjacencies = []
    input_node_text = []

    hidden_node_adjacencies = []
    hidden_node_text = []
    #print('Scaled_Activations_formatted_list', len(Scaled_Activations_formatted_list), Scaled_Activations_formatted_list)
    #print('nonScaled_Activations_formatted_list', len(nonScaled_Activations_formatted_list), nonScaled_Activations_formatted_list)
    # print('frequency_of_atoms', len(frequency_of_atoms), frequency_of_atoms)
    # print('network_graph', len(network_graph.nodes), network_graph.nodes)

    node_names = {'0': ' Carbon', '1': ' Nitrogen', '2': ' Oxygen ', '3': ' Fluorine', '4': ' Indium', '5': ' Chlorine', '6': ' Bromine'}
    first_node_of_each_layer = []
    for layer in my_nodes_name_index_layers:
        first_node_of_each_layer.append(layer[0])

    for node, adjacencies in enumerate(network_graph.adjacency()):
        if node <= 6:
            input_node_adjacencies.append(len(adjacencies[1]))
            input_node_text.append("The selected Molecule has " + str(frequency_of_atoms[node]) + node_names[str(node)] + " Atom/s")#+ str(list(network_graph.nodes.keys())[node]))
        elif node > 6 and node < first_node_of_each_layer[-2]:
            hidden_node_adjacencies.append(len(adjacencies[1]))
            #print(int(node/7)*7)
            formula = supscript(supscript(supscript(supscript(supscript(supscript('D\u0303', '-'), '1') + 'ᐟ', '2') + 'A\u0303' + supscript('D\u0303', '-'), '1') + 'ᐟ', '2') + 'H', str(first_node_of_each_layer.index(int(node/7)*7)))
            hidden_node_text.append("Neuron: " + str(list(network_graph.nodes.keys())[node - 7]) + ", <br>GCN Formula: " + "<i>" + formula + "</i>" + ", <br>Avg. non-Scaled Activations(of the Neuron for Atoms): " + str(nonScaled_Activations_formatted_list[node-7]) + ", <br>Avg. Scaled Activations(of the Neuron for Atoms): " + str(Scaled_Activations_formatted_list[node-7]))  # + " Weights: " + str(Weights[node]))
        elif node < first_node_of_each_layer[-1] and node >= first_node_of_each_layer[-2]:
            hidden_node_adjacencies.append(len(adjacencies[1]))
            gap_formula = "r" + "\u1d62" + " = " + "¹⁄ₙ" + "\u03A3\u1D62" + "\u207f" + "x" + '\u1d62'
            hidden_node_text.append("Neuron: " + str(list(network_graph.nodes.keys())[node - 7]) + ", <br>GAP Formula: " + gap_formula + ", <br>Avg. non-Scaled Activations(of the Neuron for Atoms): " + str(nonScaled_Activations_formatted_list[node - 7]) + ", <br>Avg. Scaled Activations(of the Neuron for Atoms): " + str(Scaled_Activations_formatted_list[node - 7]))  # + " Weights: " + str(Weights[node]))
        elif node >= first_node_of_each_layer[-1]:
            hidden_node_adjacencies.append(len(adjacencies[1]))
            hidden_node_text.append("Neuron: " + str(list(network_graph.nodes.keys())[node - 7]) + ", <br>Fully Connected Formula: " + ", <br>Avg. non-Scaled Activations(of the Neuron for Atoms): " + str(nonScaled_Activations_formatted_list[node - 7]) + ", <br>Avg. Scaled Activations(of the Neuron for Atoms): " + str(Scaled_Activations_formatted_list[node - 7]))

    #print(input_node_text)
    #print(hidden_node_text)
    #print("max activation: ", max(Activations),  "min activation: ", min(Activations))
    hidden_nodes_color_threshod = (max(Scaled_Activations_formatted_list)+min(Scaled_Activations_formatted_list))/2
    input_nodes_color_threshod = (max(frequency_of_atoms)+min(frequency_of_atoms))/2
    # print("thresholds done")
    # print(len(network_graph.nodes), network_graph.nodes)

    edge_text = []
    for i, edge in enumerate(network_graph.edges.values()):
        #edge_type.append(edge['type'])
        edge_text.append("edge")

    #'0': 'C', '1': 'N', '2': 'O', '3': 'F', '4': 'I', '5': 'Cl', '6': 'Br'
    input_node_trace.marker.color = [0, 1, 2, 3, 4, 5, 6]#frequency_of_atoms
    input_node_trace.text = input_node_text


    #print(my_nodes_name_index)
    hidden_node_trace.marker.color = Scaled_Activations_formatted_list
    hidden_node_trace.text = hidden_node_text

    #edge_trace.marker.color = edge_type
    #print(len(edge_type), edge_type)
    #edge_trace.line.color = edge_type
    #edge_text_trace.marker.color = edge_type
    edge_text_trace.text = edge_text
    #for i, edge in enumerate(original_graph.edges.values()):
    #    print(i, "  ", edge['type'])

    ##########################################################################     Annotations     #####################
    #####     layer names  #####
    first_node_of_each_layer = []
    for layer in my_nodes_name_index_layers:
        first_node_of_each_layer.append(layer[0])
    print(first_node_of_each_layer)

    xshift_for_layers_labels = -30
    yshift_for_layers_labels = +30


    nick_node_names = {'0': 'C', '1': 'N', '2': 'O', '3': 'F', '4': 'In', '5': 'Cl', '6': 'Br'}
    network_annotations = []
    for i in range(len(network_graph.nodes)):
        if i <= 6:
            posit = network_graph.nodes[str(i)]['pos']
            if i in first_node_of_each_layer and i != first_node_of_each_layer[-1] and i != first_node_of_each_layer[-2]:
                network_annotations.append(dict(x=posit[0], y=posit[1],
                                                text='Input Layer', xanchor='left',
                                                xshift=xshift_for_layers_labels - 65,
                                                yshift=yshift_for_layers_labels - 20,
                                                font=dict(color='black', size=12, family="Times New Roman"),
                                                showarrow=False, arrowhead=1, ax=0, ay=0
                                                )
                                           )
            if frequency_of_atoms[i] >= input_nodes_color_threshod:
                network_annotations.append(dict(x=posit[0], y=posit[1],
                                                text=nick_node_names[str(i)], xanchor='left',
                                                xshift=-7, yshift=+2, font=dict(color='white', size=15, family="Times New Roman"),
                                                showarrow=False, arrowhead=1, ax=0, ay=0
                                                )
                                           )
            elif frequency_of_atoms[i] < input_nodes_color_threshod:
                network_annotations.append(dict(x=posit[0], y=posit[1],
                                                text=nick_node_names[str(i)], xanchor='left',
                                                xshift=-7, yshift=+2, font=dict(color='black', size=15, family="Times New Roman"),
                                                showarrow=False, arrowhead=1, ax=0, ay=0
                                                )
                                           )
        else:
            posit = network_graph.nodes[str(i)]['pos']
            if i in first_node_of_each_layer and i == first_node_of_each_layer[-1]:
                network_annotations.append(dict(x=posit[0], y=posit[1],
                                                text='Fully-Connected Layer', xanchor='left',
                                                xshift=xshift_for_layers_labels - 28, yshift=yshift_for_layers_labels,
                                                font=dict(color='black', size=12, family="Times New Roman"),
                                                showarrow=False, arrowhead=1, ax=0, ay=0
                                                )
                                           )
            if i in first_node_of_each_layer and i == first_node_of_each_layer[-2]:
                network_annotations.append(dict(x=posit[0], y=posit[1],
                                                text='GAP Layer', xanchor='left',
                                                xshift=xshift_for_layers_labels, yshift=yshift_for_layers_labels,
                                                font=dict(color='black', size=12, family="Times New Roman"),
                                                showarrow=False, arrowhead=1, ax=0, ay=0
                                                )
                                           )
            if i in first_node_of_each_layer and i != first_node_of_each_layer[-1] and i != first_node_of_each_layer[-2]:
                network_annotations.append(dict(x=posit[0], y=posit[1],
                                                text='GCN Layer', xanchor='left',
                                                xshift=xshift_for_layers_labels, yshift=yshift_for_layers_labels,
                                                font=dict(color='black', size=12, family="Times New Roman"),
                                                showarrow=False, arrowhead=1, ax=0, ay=0
                                                )
                                           )
            if Scaled_Activations_formatted_list[i-7] >= hidden_nodes_color_threshod:
                if i-7 < 10:
                    network_annotations.append(dict(x=posit[0], y=posit[1],
                                                    text=str(i-7), xanchor='left',
                                                    xshift=-7, yshift=+2, font=dict(color='white', size=15, family="Times New Roman"),
                                                    showarrow=False, arrowhead=1, ax=0, ay=0
                                                    )
                                               )
                else:
                    network_annotations.append(dict(x=posit[0], y=posit[1],
                                                    text=str(i-7), xanchor='left',
                                                    xshift=-10, yshift=+2, font=dict(color='white', size=15, family="Times New Roman"),
                                                    showarrow=False, arrowhead=1, ax=0, ay=0
                                                    )
                                               )

            elif Scaled_Activations_formatted_list[i-7] < hidden_nodes_color_threshod:
                if i-7 < 10:
                    network_annotations.append(dict(x=posit[0], y=posit[1],
                                                    text=str(i-7), xanchor='left',
                                                    xshift=-7, yshift=+2, font=dict(color='black', size=15, family="Times New Roman"),
                                                    showarrow=False, arrowhead=1, ax=0, ay=0
                                                    )
                                               )
                else:
                    network_annotations.append(dict(x=posit[0], y=posit[1],
                                                    text=str(i-7), xanchor='left',
                                                    xshift=-10, yshift=+2, font=dict(color='black', size=15, family="Times New Roman"),
                                                    showarrow=False, arrowhead=1, ax=0, ay=0
                                                    )
                                               )


    return input_node_trace, hidden_node_trace, edge_trace, edge_text_trace, network_annotations



def Network_Figure_Drawing(edge_trace, edge_text_trace, input_node_trace, hidden_node_trace, net_fig_width, net_fig_height, prediction, annotations):
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )
    fig = go.Figure(data=[edge_trace, input_node_trace, hidden_node_trace], #, edge_text_trace],
                    layout=go.Layout(plot_bgcolor='white',
                                     width=net_fig_width, height=net_fig_height,
                                     title='<br>Model</b> Predicted Label: ' + str(prediction),
                                     titlefont_size=16,
                                     showlegend=False,
                                     hovermode='closest',
                                     annotations=annotations,
                                     margin=dict(b=20, l=5, r=5, t=40),
                                     scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis)),#, annotations=annotations),
                                     #annotations=[dict(showarrow=False,
                                      #                 xref="paper", yref="paper",
                                       #                x=0, y=0, xanchor='right', yanchor='top')]
                                    )
                    )
    #fig.add_annotation(x=0.6, y=.2,
    #                   text="Text annotation with arrow",
    #                   showarrow=True,
    #                   arrowhead=1)
    fig.update_layout(showlegend=False, title_font_family="Times New Roman")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange="reversed")
    #fig.show()
    return fig

#Network_datastructure_part(1000, 700)
def weights_of_model(model):
    GConv1_Layer_Weights = model.GConvs[0].lin.weight.data
    GConv2_Layer_Weights = model.GConvs[0].lin.weight.data
    GAP_Layer_Weights = ['GAP Node'] * 7
    Dense_Layer_Weights = model.ffn.weight

    GConv1_Layer_Weights = GConv1_Layer_Weights.detach().tolist()
    GConv2_Layer_Weights = GConv2_Layer_Weights.detach().tolist()
    Dense_Layer_Weights = Dense_Layer_Weights.detach().tolist()
    #print("GConv1: ", len(GConv1_Layer_Weights))
    #print("GConv2: ", len(GConv2_Layer_Weights))
    #print("GAP: ", len(GAP_Layer_Weights))
    #print("Dense: ", len(Dense_Layer_Weights))

    Weights = []
    Weights.extend(GConv1_Layer_Weights)
    Weights.extend(GConv2_Layer_Weights)
    Weights.extend(GAP_Layer_Weights)
    Weights.extend(Dense_Layer_Weights)
    return Weights

def scale_my_activations(node_features, out_pretrained, out_readout, post_conv2, post_conv1):
    node_features = node_features.detach().tolist()

    post_conv1 = [x.tolist() for x in post_conv1]

    #post_conv2 = post_conv2.detach().tolist()
    post_conv2 = [x.tolist() for x in post_conv2]
    #out_readout = out_readout.detach().tolist()
    out_readout = [x.tolist() for x in out_readout]
    #out_pretrained = out_pretrained.detach().tolist()
    out_pretrained = [x.tolist() for x in out_pretrained]

    node_features = np.sum(node_features, axis=0).tolist()
    node_features = [int(s) for s in node_features]
    post_conv1 = np.mean(post_conv1, axis=0)
    post_conv2 = np.mean(post_conv2, axis=0)
    out_readout = out_readout[0]
    #print(node_features)

    Activations = []
    #print("GAP: ", len(out_readout), out_readout)
    #print("GCN2: ", len(post_conv2), post_conv2)
    #print("GCN1: ", len(post_conv1), post_conv1)

    #Activations.extend(node_features)
    Activations.extend(post_conv1)
    Activations.extend(post_conv2)
    Activations.extend(out_readout)
    Activations.extend(out_pretrained[0])


    Scaled_Activations = torch.tensor([Activations], requires_grad=True)
    Scaled_Activations = F.log_softmax(Scaled_Activations, dim=1)
    Scaled_Activations = F.softmax(Scaled_Activations, dim=1)

    Scaled_Activations = Scaled_Activations.detach().tolist()[0]
    #print(len(Scaled_Activations), Scaled_Activations)

    nonScaled_Activations_formatted_list = [int(elem*10000)/10000 for elem in Activations]
    Scaled_Activations_formatted_list = [int(elem*10000)/10000 for elem in Scaled_Activations]
    #print("activations", Activations)
    #print("limited activations: ", Activations_formatted_list)
    #print("scaled activations: ", Scaled_Activations)
    #print("limited scaled activations: ", Scaled_Activations_formatted_list)

    return Scaled_Activations_formatted_list, nonScaled_Activations_formatted_list, node_features


@app.callback([Output(component_id='Visualize_Model', component_property='children'),
               Output(component_id='my_network_display_toggle', component_property='on')],
              [Input(component_id='my_network_display_toggle', component_property='on'),
               Input(component_id='Task_Dropdown', component_property='value'),
               Input(component_id='Index_Dropdown', component_property='value'),
               Input(component_id='Model_Dropdown', component_property='value'),
               Input(component_id='Method_Dropdown', component_property='value')])
def Network_display_my_figure(on, selected_task, selected_index, selected_model, selected_method):


    print("Interacttion on Network Display", on, selected_task, selected_index, selected_model, selected_method)
    #Explainability_name = 'SA'
    #Task_name = 'Graph Classification'
    load_index = 200

    if on and (selected_task == "GC") and (selected_index != None) and (selected_model != "Please Select The Task at First...") and (selected_model != "Please select the Model for Graph Classification") and (selected_model != None): # and (selected_method != "Please select the Method for Graph Classification") and (selected_method != "Please Select The Task at First...") and (selected_method != None):
        on = True
        if (selected_method == None) or (selected_method != "Please select the Method for Graph Classification") or (selected_method != "Please Select The Task at First..."):
            my_model_pretrained, optimizer, epochs = loading_pretrained_model(load_index, "SA", selected_model, 'Graph Classification')
        else:
            print("Start")
            my_model_pretrained, optimizer, epochs = loading_pretrained_model(load_index, selected_method, selected_model, 'Graph Classification')
            print("End")
        #print("PreTrained Model Loaded")
        Graph = Graphs_Family[selected_index]

        Output_of_Hidden_Layers, out_readout, ffn_output, out_pretrained_softed = my_model_pretrained(Graph)
        post_conv1 = Output_of_Hidden_Layers[0]
        post_conv2 = Output_of_Hidden_Layers[1]
        #logits = F.log_softmax(out_pretrained, dim=1)
        #prob = F.softmax(logits, dim=1)
        print("post_conv1, post_conv2, out_readout, out_pretrained = my_model_pretrained(Graph)")
        Scaled_Activations_formatted_list, nonScaled_Activations_formatted_list, frequency_of_atoms = scale_my_activations(Graph.x, out_pretrained_softed, out_readout, post_conv2, post_conv1)


        #print(on, selected_task, selected_index, selected_model, selected_method)
        print("Output of the Model for Seleceted Graph: ", out_pretrained_softed)
        print("Prob of the Model for Seleceted Graph: ", out_pretrained_softed, out_pretrained_softed.argmax(dim=1).detach().tolist())

        #print("GAP: ", len(out_readout), out_readout)
        #print("GCN2: ", len(post_conv2), post_conv2)
        #print("GCN1: ", len(post_conv1), post_conv1)


        #print(np.shape(Activations))
        StateDict = my_model_pretrained.state_dict()
        print(StateDict.keys())
        Weights = weights_of_model(my_model_pretrained)
        #print(Weights)
        print("out_pretrained_softed.argmax(dim=1).detach().tolist()[0]: ", out_pretrained_softed.argmax(dim=1).detach().tolist()[0])


        fig_network = Network_datastructure_part(Scaled_Activations_formatted_list, nonScaled_Activations_formatted_list, frequency_of_atoms, Weights, out_pretrained_softed.argmax(dim=1).detach().tolist()[0], net_fig_width, net_fig_height)
        dcc.Graph(figure=fig_network)
        return [dcc.Graph(figure=fig_network)], on

    #Graph = Mutag_Dataset[0]
    #print(Graph.x)
    #fig = My_DataStructure(Graph, fig_width, fig_height)
    #tips = px.data.tips()


    else:
        on = False
        axis = dict(showbackground=False,
                    showline=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    title=''
                    )
        fig = go.Figure(data=[],
                        layout=go.Layout(plot_bgcolor='white',
                                         width=net_fig_width, height=net_fig_height,
                                         title='<b>Menu to Select...</b>',
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
        fig.update_layout(title_font_family="Times New Roman")
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        return [dcc.Graph(figure=fig)], on



##############################################################################     General Statistics     ######################




def load_general_statistics(selected_task, selected_method):
    df = pandas.read_csv("/Users/EY33JW/PycharmProjects/pythonProject2/Comparisons_ExMethods_Final_Format.csv")
    #print(selected_task, selected_method, "===================================================")

    names_dict_for_csv_stats = {"SA": "SA", "GuidedBP": "GuidedBP", "GC": "GC", "NC": "NC"}
    #print(names_dict_for_csv_stats[str(selected_method)])
    df = df.loc[(df["Explicability Method Name"] == selected_method) & (df["Task"] == selected_task)]
    #print(df.columns)
    print("df now: ", selected_method, df)
    columns = [{'name': col, 'id': col} for col in df.columns]
    data = df.to_dict(orient='records')



    dataframe_general_classifier = df[['Task', 'DataSet', 'Model', 'AUC-ROC', 'AUC-PRC', 'Training Accuracy', "Test Accuracy", 'Avg. Training Time on 200 Epochs']]
    dataframe_general_classifier = dataframe_general_classifier.rename(columns={'Avg. Training Time on 200 Epochs': 'Avg. Training Time Per Epoch [Sec]'})
    rounded_dataframe_general_classifier = dataframe_general_classifier.round(decimals=2)
    columns_renaming_dict = \
        {
            'Avg. Training Time on 200 Epochs': 'Avg. Training Time on Epochs',
            'Test Accuracy(count, TP+TN)': 'Test Accuracy (TP+TN)',
        }

    # call rename () method
    rounded_dataframe_general_classifier.rename(columns=columns_renaming_dict, inplace=True)



    columns_general_classifier = [{'name': col, 'id': col} for col in rounded_dataframe_general_classifier.columns]
    data_general_classifier = rounded_dataframe_general_classifier.to_dict(orient='records')

    #print(dataframe_general_classifier.iloc[0])
    #dataframe_general_classifier.loc[0, ['AUC-ROC']] = int(dataframe_general_classifier.loc[0, ['AUC-ROC']] * 10000) / 10000
    #dataframe_general_classifier.loc[0, ['AUC-PRC']] = int(dataframe_general_classifier.loc[0, ['AUC-PRC']] * 10000) / 10000
    #dataframe_general_classifier.loc[0, ['Accuracy']] = int(dataframe_general_classifier.loc[0, ['Accuracy']] * 10000) / 10000
    #dataframe_general_classifier.loc[0, ['Avg. Training Time on 200 Epochs']] = int(dataframe_general_classifier.loc[0, ['Avg. Training Time on 200 Epochs']] * 10000) / 10000
    #print(dataframe_general_classifier.iloc[0])


    dataframe_general_explainer = df[
        ['Explicability Method Name', 'Fidelity', 'Contrastivity', 'Sparsity', 'Saliency Map Generation Time']]
    dataframe_general_explainer = dataframe_general_explainer.rename(
        columns={'Saliency Map Generation Time': 'Saliency Map Generation Time [Sec]'})
    rounded_dataframe_general_explainer = dataframe_general_explainer.round(decimals=2)

    columns_general_explainer = [{'name': col, 'id': col} for col in rounded_dataframe_general_explainer.columns]
    data_general_explainer = rounded_dataframe_general_explainer.to_dict(orient='records')
    #print(dataframe_general_explainer.iloc[0])
    #dataframe_general_explainer.loc[0, ['Fidelity']] = int(dataframe_general_explainer.loc[0, ['Fidelity']] * 10000) / 10000
    #dataframe_general_explainer.loc[0, ['Contrastivity']] = int(dataframe_general_explainer.loc[0, ['Contrastivity']] * 10000) / 10000
    #dataframe_general_explainer.loc[0, ['Sparsity']] = int(dataframe_general_explainer.loc[0, ['Sparsity']] * 10000) / 10000
    #dataframe_general_explainer.loc[0, ['Saliency Map Generation Time']] = int(dataframe_general_explainer.loc[0, ['Saliency Map Generation Time']] * 10000) / 10000
    #print(dataframe_general_explainer.iloc[0])
    return data_general_classifier, columns_general_classifier, data_general_explainer, columns_general_explainer


@app.callback([Output(component_id='my_model_and_method_general_statistics_display_toggle', component_property='on'),
               Output(component_id='Visualize_General_Performance_of_Classifier', component_property='data'),
               Output(component_id='Visualize_General_Performance_of_Classifier', component_property='columns'),
               Output(component_id='Visualize_General_Performance_of_Explainer', component_property='data'),
               Output(component_id='Visualize_General_Performance_of_Explainer', component_property='columns')],
              [Input(component_id='my_model_and_method_general_statistics_display_toggle', component_property='on'),
               Input(component_id='Task_Dropdown', component_property='value'),
               Input(component_id='Model_Dropdown', component_property='value'),
               Input(component_id='Method_Dropdown', component_property='value')])
def display_general_statistics_of_the_selected_model_and_method(on, selected_task, selected_model, selected_method):
    #print("The Third Display: ", on, selected_task, selected_model, selected_method)
    if on and (selected_task == "GC") and (selected_model != "Please Select The Task at First...") and (selected_model != "Please select the Model for Graph Classification") and (selected_model != None) and (selected_method != "Please select the Method for Graph Classification") and (selected_method != "Please Select The Task at First...") and (selected_method != None):
        on = True
        data_general_classifier, columns_general_classifier, data_general_explainer, columns_general_explainer = load_general_statistics(selected_task, selected_method)
        #, data_general_explaienr, data_general_explaienr
        return on, data_general_classifier, columns_general_classifier, data_general_explainer, columns_general_explainer
    else:
        on = False
        return on, None, None, None, None

#############################################################################    Instance Specific Stats    ############

def generate_instance_specific_statistics(selected_task, selected_index, selected_model, selected_method, importance_threshold):


    #############################################################################################    CLASSIFIER    #####
    # print(selected_task, selected_index, selected_model, selected_method)


    task_dict = {"GC": "Graph Classification", "NC": "Node Classification"}
    my_model_pretrained, optimizer, epochs = loading_pretrained_model(200, selected_method, selected_model, task_dict[selected_task])
    # print("PreTrained Model Loaded")
    print("In Instance Specific Statistics: ", selected_index)
    Graph = Graphs_Family[selected_index]

    starting_time = perf_counter()
    post_conv1, post_conv2, out_readout, out_pretrained = my_model_pretrained(Graph)
    #logits = F.log_softmax(out_pretrained, dim=1)
    #prob = F.softmax(logits, dim=1)
    taken_time = perf_counter() - starting_time


    #print("this one: ", prob, prob.detach().tolist()[0], "winner index: ", prob.argmax(dim=1).detach().tolist()[0], "taken time: ", taken_time, Graph.y)
    predicted_index = out_pretrained.argmax(dim=1).detach().tolist()
    winning_prob = [out_pretrained.detach().tolist()[0][out_pretrained.argmax(dim=1).detach().tolist()[0]]]
    real_index = Graph.y.detach().tolist()
    consumed_time = [taken_time]
    #print("predicted_index: ", predicted_index, "winning_prob: ", winning_prob, "real_index: ", real_index, "consumed_time: ", consumed_time)

    instance_specific_classification_statistics = pandas.DataFrame({'Predicted Label': predicted_index, 'Probability': winning_prob, 'Real Label': real_index, 'Classification Time': consumed_time})
    instance_specific_classification_statistics = instance_specific_classification_statistics.rename(
        columns={'Classification Time': 'Classification Time [Sec]'})
    #print(instance_specific_classification_statistics)
    instance_specific_classification_statistics = instance_specific_classification_statistics.round(decimals=4)
    #print(instance_specific_classification_statistics)
    columns_instance_specific_classifier = [{'name': col, 'id': col} for col in instance_specific_classification_statistics.columns]
    data_instance_specific_classifier = instance_specific_classification_statistics.to_dict(orient='records')



    ##############################################################################################    EXPLAINER    #####
    if selected_method == "SA":
        new_output = SA_Method_as_Class.SA_GC(task="Graph Classification", method=selected_method, model_name=selected_model,
                                              graph=[Graph], importance_threshold=importance_threshold, load_index=200,
                                              input_dim=len(Graph.x[0]), hid_dim=7, output_dim=2, normalize_coeff=1000000)
    elif selected_method == "GuidedBP":
        new_output = GuidedBP_Method_as_Class.GuidedBP_GC(task="Graph Classification", method=selected_method, model_name=selected_model,
                                                          graph=[Graph], importance_threshold=importance_threshold, load_index=200,
                                                          input_dim=len(Graph.x[0]), hid_dim=7, output_dim=2, normalize_coeff=10)
    elif selected_method == "CAM":
        new_output = CAM_Method_as_Class.CAM_GC(task="Graph Classification", method=selected_method, model_name=selected_model,
                                                graph=[Graph], importance_threshold=importance_threshold, load_index=200,
                                                input_dim=len(Graph.x[0]), hid_dim=7, output_dim=2, normalize_coeff=10,
                                                DataSet_name = "MUTAG")
    elif selected_method == "Grad-CAM":
        new_output = CAM_Method_as_Class.CAM_GC(task="Graph Classification", method=selected_method, model_name=selected_model,
                                                graph=[Graph], importance_threshold=importance_threshold, load_index=200,
                                                input_dim=len(Graph.x[0]), hid_dim=7, output_dim=2, normalize_coeff=100,
                                                DataSet_name = "MUTAG")
    elif selected_method == "LRP":
        new_output = LRP_Method_as_Class.LRP_GC(task="Graph Classification", method=selected_method, model_name=selected_model,
                                                graph=[Graph], importance_threshold=importance_threshold, load_index=200,
                                                input_dim=len(Graph.x[0]), hid_dim=7, output_dim=2, normalize_coeff=100,
                                                DataSet_name="MUTAG")
    elif selected_method == "ExcitationBP":
        new_output = LRP_Method_as_Class.LRP_GC(task="Graph Classification", method=selected_method, model_name=selected_model,
                                                graph=[Graph], importance_threshold=importance_threshold, load_index=200,
                                                input_dim=len(Graph.x[0]), hid_dim=7, output_dim=2, normalize_coeff=100,
                                                DataSet_name="MUTAG")
    elif selected_method == "GNNExplainer":
        new_output = GNNExplainer_Method_as_Class.GNNExplainer_GC(task="Graph Classification", method="GNNExplainer", model_name=selected_model,
                                                                  graph=[Graph], importance_threshold=importance_threshold, load_index=200,
                                                                  input_dim=7, hid_dim=7, output_dim=2, normalize_coeff=100,
                                                                  DataSet_name="MUTAG")

    elif selected_method == "PGExplainer":
        new_output = PGExplainer_Method_as_Class.PGExplainer(Model_Name=selected_model, Explainability_name='PGExplainer',
                                                             Task_name='Graph Classification', classifier_load_index=200,
                                                             explainer_save_index=10000, Exp_Epoch=100, Exp_lr=0.001,
                                                             input_dim=len(Graph.x[0]), hid_dim=7, output_dim=2,
                                                             importance_threshold=importance_threshold, ExTrain_or_ExTest="test",
                                                             Exp_Load_index=100, your_dataset=[Graph], target_class="correct",
                                                             DataSet_name="MUTAG", classifier_save_index=200)
    elif selected_method == "GraphMask":
        new_output = GraphMask_Method_as_Class.GraphMask(Model_Name=selected_model, Explainability_name='GraphMask',
                                                         Task_name='Graph Classification', classifier_load_index=200,
                                                         explainer_save_index=50, Exp_Epoch=50, Exp_lr=0.001,
                                                         explainer_hid_dim=7, input_dim=len(Graph.x[0]), hid_dim=7,
                                                         output_dim=2, DataSet_name="MUTAG",
                                                         importance_threshold=importance_threshold,
                                                         Exp_Load_index=100, ExTrain_or_ExTest="test",
                                                         your_dataset=[Graph],
                                                         target_class="correct")
    elif selected_method == "SubgraphX":
        new_output = SubGraphX_offline_Method_as_Class.SubGraphX_off_the_fly(your_dataset=Graph,
                                                                             Task_name='Graph Classification',
                                                                             Model_Name="GCN_plus_GAP",
                                                                             classifier_load_index=200,
                                                                             input_dim=7, hid_dim=7, output_dim=2,
                                                                             loading_graph_index=selected_index+1,
                                                                             category='correct', DataSet_name="MUTAG")
    elif selected_method == "PGMExplainer":
        new_output = PGMExplainer_Method_as_Class.PGM_Graph_Explainer(Model_Name="GCN_plus_GAP",
                                                                      classifier_load_index=200,
                                                                      input_dim=7, hid_dim=7, output_dim=2, graph=Graph,
                                                                      DataSet_name="MUTAG", perturb_feature_list=[None],
                                                                      perturb_mode="mean", perturb_indicator="abs",
                                                                      importance_threshold=importance_threshold)
    elif selected_method == "CF\u00b2":
        new_output = CF2Explainer_Method_as_Class.CF2_Explaination(Model_Name="GCN_plus_GAP", your_dataset=[Graph],
                                                                   explainer_epochs=1000, fix_exp=None,
                                                                   classifier_load_index=200, DataSet_name="MUTAG",
                                                                   input_dim=7, hid_dim=7, output_dim=2,
                                                                   importance_threshold=importance_threshold)





    instance_specific_explanation_statistics = pandas.DataFrame({'Explanation Time [Sec]': [new_output.it_took]})
    instance_specific_explanation_statistics = instance_specific_explanation_statistics.round(decimals=5)
    columns_instance_specific_explainer = [{'name': col, 'id': col} for col in instance_specific_explanation_statistics.columns]
    data_instance_specific_explainer = instance_specific_explanation_statistics.to_dict(orient='records')



    return data_instance_specific_classifier, columns_instance_specific_classifier, data_instance_specific_explainer, columns_instance_specific_explainer



@app.callback([Output(component_id='my_model_and_method_instance_specific_statistics_display_toggle', component_property='on'),
               Output(component_id='Visualize_Instance_Specific_Performance_of_Classifier', component_property='data'),
               Output(component_id='Visualize_Instance_Specific_Performance_of_Classifier', component_property='columns'),
               Output(component_id='Visualize_Instance_Specific_Performance_of_Explainer', component_property='data'),
               Output(component_id='Visualize_Instance_Specific_Performance_of_Explainer', component_property='columns')],
              [Input(component_id='my_model_and_method_instance_specific_statistics_display_toggle', component_property='on'),
               Input(component_id='Task_Dropdown', component_property='value'),
               Input(component_id='Index_Dropdown', component_property='value'),
               Input(component_id='Model_Dropdown', component_property='value'),
               Input(component_id='Method_Dropdown', component_property='value'),
               Input(component_id='importance_threshold_my-slider', component_property='value')])
def display_instance_specific_statistics_of_the_selected_model_and_method(on, selected_task, selected_index, selected_model, selected_method, importance_threshold):
    print("Importance Threshold: ", importance_threshold)
    if on and (selected_task == "GC") and (selected_index != "Please Select The Task at First...") and (selected_index != "Please select the Graph") and (selected_index != None) and (selected_model != "Please Select The Task at First...") and (selected_model != "Please select the Model for Graph Classification") and (selected_model != None) and (selected_method != "Please select the Method for Graph Classification") and (selected_method != "Please Select The Task at First...") and (selected_method != None):
        on = True
        data_instance_specific_classifier, columns_instance_specific_classifier, data_instance_specific_explainer, columns_instance_specific_explainer = generate_instance_specific_statistics(selected_task, selected_index, selected_model, selected_method, importance_threshold)

        return on, data_instance_specific_classifier, columns_instance_specific_classifier, data_instance_specific_explainer, columns_instance_specific_explainer
    else:
        on = False
        return on, None, None, None, None


if __name__ == '__main__':
    app.run_server(debug=True)


