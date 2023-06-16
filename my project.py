import os
os.environ['KMP_DUPLICATE_LIB_OK']= 'True'
import tkinter as tk
from tkinter import messagebox
import torch
import torch_geometric
from torch_geometric.datasets import TUDataset
import pyvis
from pyvis.network import Network
from math import sin, cos
import networkx as nx
from pyvis.network import Network
from urllib.request import urlopen
import tkinterhtml as th
import webview
from tkinterweb import HtmlFrame
import urllib
import dash
from dash import html
from dash import dcc
import plotly.graph_objects as go
import plotly.express as px





#
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class App():
    def __init__(self):

        self.window = tk.Tk()
        self.window.title("Tk Example")
        self.window.configure(padx=10, pady=20, background="#D3D3D3")
        self.window.title('GUI for Explicability Methods')
        #self.window.maxsize(900, 600)
        self.window.geometry("1512x900")
        self.dataset = TUDataset(root='data/TUDataset', name='MUTAG')
        #print(self.dataset[0].x)

#######------FRAMES-----###########################################
        self.Main_Vis_Frame = tk.Frame(self.window, width=1490, height=800, bg='#EBECF0', pady=10, padx=10)
        self.Main_Vis_Frame.grid()

        self.Buttons_Vis_Frame = tk.Frame(self.Main_Vis_Frame, padx=0, pady=0, bg='#F3F6FB')
        self.Buttons_Vis_Frame.place(x=10, y=10)

        self.Display_Frame = tk.Frame(self.Main_Vis_Frame, bg='#EEEEFF', width=1230, height=500, pady=0, padx=0)
        self.Display_Frame.place(x=240, y=0)

#######----Canvas Frame----##############################

        #self.my_canvas = tk.Canvas(self.Display_Frame, width=1230, height=500)
        #self.my_canvas.pack()
        #self.my_polygon = self.round_rectangle(10, 10, 1230, 500, radius=25, fill="#EECCEE") #"#EEEEFF"
        #self.my_canvas.moveto(self.my_polygon, 10, 10)
        #self.f = plt.Figure(figsize=(5, 5), dpi=300)
        #self.a = urlopen("/nodes.html")
        #self.r = self.a.read()
        #self.d = self.r.decode()

        #self.html = th.HtmlFrame(self.Display_Frame)
        #self.html.pack()
        #self.html.set_content(self.d)
        #self.folder = os.path.realpath(r'/Users/EY33JW/PycharmProjects/pythonProject1/')
        ###self.filelist = [fname for fname in os.listdir(self.folder)]
        #webview.create_window('Geeks for Geeks', '/Users/EY33JW/PycharmProjects/pythonProject1/nodes.html')
        #webview.screens()


        ###self.frame = HtmlFrame(self.Display_Frame, horizontal_scrollbar="auto")  # create HTML browser
        ###self.frame.set_content(urllib.request.urlopen("/Users/EY33JW/PycharmProjects/pythonProject1/nodes.html").read().decode())

        ###self.frame.load_website("/Users/EY33JW/PycharmProjects/pythonProject1/nodes.html")  # load a website
        ###self.frame.pack(fill="both", expand=True)
        #self.web_frame = HtmlFrame(self.Display_Frame)  # create HTML browser

        #self.web_frame.load_website("/Users/EY33JW/PycharmProjects/pythonProject1/nodes.html")  # load a website
        #self.web_frame.pack(fill="both", expand=True)


        #self.canvas = FigureCanvasTkAgg(nx.draw(self.G), master=self.Display_Frame)
        #self.canvas.draw()
        #self.canvas.get_tk_widget().place(x=240, y=0)


########----Task----################################################

        self.Task_Options = ["GC", "NC"]
        self.Task_Default_Options = ["Select the Task..."]

        self.Task_Status = tk.StringVar(self.Buttons_Vis_Frame)
        self.Task_Status.set(self.Task_Default_Options[0])
        self.Task_Status.trace('w', self.reset_before_contradiction)

        self.Task_DropDown = tk.OptionMenu(self.Buttons_Vis_Frame, self.Task_Status, *self.Task_Options, command=self.update_option_menu)
        self.Task_DropDown.config(width=20, height=1)
        self.Task_DropDown.grid(column=0, row=0)

########----Query----################################################

        self.Query_Options_for_Graph = ["Graph Name", "Graph Index"]
        self.Query_Options_for_Node = ["Node Name", "Node Index"]
        self.Query_Default_Options = ["Please Select the Task First..."]

        self.Query_Status = tk.StringVar(master=self.Buttons_Vis_Frame)
        self.Query_Status.set(self.Query_Default_Options[0])
        self.Query_Status.trace('w', self.option_select)

        self.Query_DropDown = tk.OptionMenu(self.Buttons_Vis_Frame, self.Query_Status, *self.Query_Default_Options[0])
        self.Query_DropDown.config(width=20, height=1)
        self.Query_DropDown.grid(column=0, row=1)
        #self.Query_DropDown.place(x=0, y=24)
        #self.Query_DropDown.config(width=19)

########----Model----################################################

        self.Model_Options = ["GCN + GAP", "GCNND", "DGCNN", "DIFFPOOL", "DIFFPOOLD", "GraphSig + DNN"]
        self.Model_Default_Options = ["Model Config."]

        self.Model_Status = tk.StringVar(master=self.Buttons_Vis_Frame)
        self.Model_Status.set(self.Query_Default_Options[0])
        self.Model_Status.trace('w', self.option_select)

        self.Model_DropDown = tk.OptionMenu(self.Buttons_Vis_Frame, self.Model_Status, *self.Model_Options)
        self.Model_DropDown.config(width=20, height=1)
        self.Model_DropDown.grid(column=0, row=2)
        #self.Model_DropDown.place(x=0, y=48)
        #self.Model_DropDown.config(width=19)

########----Method----################################################

        self.Method_Options_for_Graph = ['SA', 'Guided BP', 'CAM', 'Grad-CAM', 'SubgraphX', 'LRP', 'Excitation BP']
        self.Method_Options_for_Node = ['SA', 'Guided BP', 'CAM', 'Grad-CAM', 'SubgraphX', 'LRP', 'Excitation BP', 'GraphLime']
        self.Method_Default_Options = ["Please Select the Task First..."]

        self.Method_Status = tk.StringVar(master=self.Buttons_Vis_Frame)
        self.Method_Status.set(self.Query_Default_Options[0])
        self.Method_Status.trace('w', self.option_select)

        self.Method_DropDown = tk.OptionMenu(self.Buttons_Vis_Frame, self.Method_Status, *self.Method_Default_Options[0])
        self.Method_DropDown.config(width=20, height=1)
        self.Method_DropDown.grid(column=0, row=3)
        #self.Method_DropDown.place(x=0, y=72)
        #self.Method_DropDown.config(width=19)

########----OutPut Dictionary----################################################

        self.my_output_dict = {"Task": [], "Index_OR_Name": [], "Model": [], "Explainable Method": []}

########----Start Button----################################################

        self.Start_Button = tk.Button(self.Buttons_Vis_Frame, text='Click to Start...', command=self.start_button_pressed)
        self.Start_Button.config(width=21, height=1)
        self.Start_Button.grid(column=0, row=4)
        #self.Start_Button.place(x=0, y=96)
        #self.Start_Button.config(width=20)

########----ReStart Button----################################################

        self.ReStart_Button = tk.Button(self.Buttons_Vis_Frame, text='Click to ReSet Defaults...', command=self.restart_button_pressed)
        self.ReStart_Button.config(width=21, height=1)
        self.ReStart_Button.grid(column=0, row=5)
        #self.ReStart_Button.place(x=0, y=120)
        #self.ReStart_Button.config(width=20)

#######----Display Frame____#############################
        '''self.Ex_Options = ["GC", "NC"]
        self.Ex_Default_Options = ["position me"]

        self.Ex_Status = tk.StringVar(self.Display_Frame)
        self.Ex_Status.set(self.Task_Default_Options[0])


        self.Ex_DropDown = tk.OptionMenu(self.Display_Frame, self.Task_Status, *self.Task_Options)
        self.Ex_DropDown.place(x=0, y=0)'''

        #self.frame = HtmlFrame(self.Display_Frame)  # create HTML browser

        #self.frame.load_website("/Users/EY33JW/PycharmProjects/pythonProject1/nodes.html")  # load a website
        #self.frame.pack(fill="both", expand=True)  # attach the HtmlFrame widget to the parent window

        self.window.mainloop()

    def round_rectangle(self, x1, y1, x2, y2, radius, **kwargs):
        points = [x1 + radius, y1,
                  x1 + radius, y1,
                  x2 - radius, y1,
                  x2 - radius, y1,
                  x2, y1,
                  x2, y1 + radius,
                  x2, y1 + radius,
                  x2, y2 - radius,
                  x2, y2 - radius,
                  x2, y2,
                  x2 - radius, y2,
                  x2 - radius, y2,
                  x1 + radius, y2,
                  x1 + radius, y2,
                  x1, y2,
                  x1, y2 - radius,
                  x1, y2 - radius,
                  x1, y1 + radius,
                  x1, y1 + radius,
                  x1, y1]
        return self.my_canvas.create_polygon(points, **kwargs, smooth=True)

    def restart_button_pressed(self, *args):
        self.Task_Status.set(self.Task_Default_Options[0])
        self.Query_Status.set(self.Query_Default_Options[0])
        self.Model_Status.set(self.Query_Default_Options[0])
        self.Method_Status.set(self.Query_Default_Options[0])
        self.my_output_dict["Task"] = []
        self.my_output_dict["Index_OR_Name"] = []
        self.my_output_dict["Model"] = []
        self.my_output_dict["Explainable Method"] = []

    def start_button_pressed(self,*args):
        if self.Task_Status.get() != self.Task_Default_Options[0] and self.Query_Status.get() != self.Query_Default_Options[0] and self.Model_Status.get() != self.Model_Default_Options[0] and self.Method_Status.get() != self.Method_Default_Options[0]:
            self.my_output_dict["Task"].append(self.Task_Status.get())
            self.my_output_dict["Index_OR_Name"].append(self.Query_Status.get())
            self.my_output_dict["Model"].append(self.Model_Status.get())
            self.my_output_dict["Explainable Method"].append(self.Method_Status.get())
            self.option_select()
        elif self.Task_Status.get() == self.Task_Default_Options[0]:
            messagebox.showerror('Python Error', 'Error: Please Select a Task')
        elif self.Query_Status.get() == self.Query_Default_Options[0]:
            messagebox.showerror('Python Error', 'Error: Please Select a Query')
        elif self.Model_Status.get() == self.Model_Default_Options[0]:
            messagebox.showerror('Python Error', 'Error: Please Select a Model')
        elif self.Method_Status.get() == self.Method_Default_Options[0]:
            messagebox.showerror('Python Error', 'Error: Please Select a Method')

    def update_option_menu(self, *args):
        if self.Task_Status.get() == "GC":
            #self.add_option
            menu = self.Query_DropDown["menu"]
            menu.delete(0, "end")
            for string in self.Query_Options_for_Graph:
                menu.add_command(label=string, command=lambda value=string: self.Query_Status.set(value))

            menu2 = self.Method_DropDown["menu"]
            menu2.delete(0, "end")
            for string in self.Method_Options_for_Graph:
                menu2.add_command(label=string, command=lambda value=string: self.Method_Status.set(value))
        elif self.Task_Status.get() == "NC":
            #self.add_option
            menu = self.Query_DropDown["menu"]
            menu.delete(0, "end")
            for string in self.Query_Options_for_Node:
                menu.add_command(label=string, command=lambda value=string: self.Query_Status.set(value))

            menu2 = self.Method_DropDown["menu"]
            menu2.delete(0, "end")
            for string in self.Method_Options_for_Node:
                menu2.add_command(label=string, command=lambda value=string: self.Method_Status.set(value))

    def reset_before_contradiction(self, *args):

        self.Query_Status.set(self.Query_Default_Options[0])
        self.Model_Status.set(self.Query_Default_Options[0])
        self.Method_Status.set(self.Query_Default_Options[0])

        self.my_output_dict["Index_OR_Name"] = []
        self.my_output_dict["Model"] = []
        self.my_output_dict["Explainable Method"] = []
        self.option_select()
    def option_select(self, *args):
        #print(self.Task_Status.get())
        print(self.my_output_dict)




App()

