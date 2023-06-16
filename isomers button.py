import dash
from dash import Dash, dcc, html, Input, Output, State, dash_table
from dash.dash_table.Format import Group, Format, Scheme

import dash_bootstrap_components as dbc
from dash import Dash, html
from dash_iconify import DashIconify

FA = "https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/css/bootstrap.min.css"

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, FA])

app.layout = dbc.Button([html.I(className="glyphicon glyphicon-menu-left"), ' shift ',html.I(className="glyphicon glyphicon-menu-right")])



if __name__ == "__main__":
    app.run_server(debug=True)