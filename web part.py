import dash
from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import dash_daq as daq



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

controls2 = dbc.Card(
    [
        dbc.CardHeader("Checking", style={'text-align': 'center', 'padding': '0px',
                                                                           "font-weight": "bold", "height": "23px",
                                                                           'backgroundColor': '#e0e0e0',
                                                                           'margin-bottom': '2px',
                                                                           'font-family': 'Times New Roman'}),
        #html.Div(id='GUI_Input', style={'textAlign': 'left'}, children=[
            #html.H1(children='GUI for XAI on Graph-Structured Data',
            #        style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 40, 'color': 'red'}),
            # Create a title with H1 tag

            #     html.Div(children='hello',
                  #         style={'textAlign': 'left', 'color': 'grey'}),  # Display some text
        daq.BooleanSwitch(id="my_display_toggle2", style={"width": "150px", 'font-family': 'Times New Roman', 'margin-top': '10px', 'margin-left': '10px', 'margin-right': '110px'}, labelPosition='right', on=False, color="red", label={"label": "Graph Display", 'style': {'margin-left': '15px'}}),
            #daq.BooleanSwitch(id="my_network_display_toggle2", style={"width": "300px", 'font-family': 'Times New Roman', 'margin-top': '10px', 'margin-left': '10px'}, labelPosition='left', on=False, color="red", label={"label": "Neural Network Display", 'style': {'margin-left': '15px'}}),
            #daq.BooleanSwitch(id="my_model_and_method_general_statistics_display_toggle2", style={"width": "300px", 'font-family': 'Times New Roman', 'margin-top': '10px', 'margin-left': '10px'}, labelPosition='left', on=False, color="red", label={"label": "Model and Method General Performance Display", 'style': {'margin-left': '15px'}}),
            #daq.BooleanSwitch(id="my_model_and_method_instance_specific_statistics_display_toggle2", style={"width": "300px", 'font-family': 'Times New Roman', 'margin-top': '10px', 'margin-left': '10px'}, labelPosition='left', on=False, color="red", label={"label": "Model and Method Instance-Specific Performance Display", 'style': {'margin-left': '15px'}}),
        #]
                 #),
    ],
    body=True, style={'margin-right': '10px'}
)
left_margin = 5
right_margin = 5


app.layout = dbc.Container(
    [
        html.H1("GUI for XAI on Graph-Structured Data", style={'textAlign': 'center', 'marginTop': 10, 'marginBottom': 10, 'color': 'grey'}),
        html.Hr(),
        dbc.Col(
            [
                dbc.Row(dbc.Card([html.Div("Controls")], body=True), align="start", style={'margin-right': right_margin, 'margin-left': left_margin, 'margin-bottom': 5}),

                                #######dbc.Card(html.Div("Second Column"), body=True)

                dbc.Row(dbc.Card([html.Div("Tables")], body=True), align="start", style={'margin-right': right_margin, 'margin-left': left_margin, 'margin-bottom': 5})
            ], align="start", width=2, md=4
                ),
        dbc.Col(
            [

                dbc.Row(dbc.Card(html.Div("Graph DIsplay"),body=True), align="start",
                        style={'margin-right': right_margin, 'margin-left': left_margin, 'margin-bottom': left_margin}),

                #######dbc.Card(html.Div("Second Column"), body=True)

                dbc.Row(dbc.Card(html.Div("Network DIsplay"), body=True), align="start",
                        style={'margin-right': right_margin, 'margin-left': left_margin, 'margin-bottom': 5})
            ], align="start", width=4, md=8
        ),
    ]
)


if __name__ == '__main__':
    app.run_server(debug=True)