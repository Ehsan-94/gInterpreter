import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, html

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Link", href="#")),
        dbc.DropdownMenu(
            nav=True,
            in_navbar=True,
            label="Menu",
            children=[
                dbc.DropdownMenuItem("Entry 1"),
                dbc.DropdownMenuItem("Entry 2"),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem("Entry 3"),
            ],
        ),
    ],
    brand="Demo",
    brand_href="#",
    sticky="top",
)
tr_style = {"writingMode": "vertical-lr"}
table_style = {'borderWidth': '10px', "textAlign":"center"}
# table_style.update(tr_style)
table_header = [
    html.Thead(
        html.Tr(
            [
                # html.Th("First Name", className="verticalTableHeader"),
                # html.Th("Last Name",  className="verticalTableHeader")
                html.Th("First Name"),
                html.Th("Last Name")
            ]
        # style={"verticalAlign": "bottom"}
        ),
    )
]

row1 = html.Tr([html.Td("Arthur"), html.Td("Dent")])
row2 = html.Tr([html.Td("Ford"), html.Td("Prefect")])
row3 = html.Tr([html.Td("Zaphod"), html.Td("Beeblebrox")])
row4 = html.Tr([html.Td("Trillian"), html.Td("Astra")])

table_body = [html.Tbody([row1, row2, row3, row4])]

df = pd.DataFrame(
    {
        "First Name": ["Arthur", "Ford", "Zaphod", "Trillian"],
        "Last Name": ["Dent", "Prefect", "Beeblebrox", "Astra"],
    }
)

body = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Heading"),
                        html.P(
                            """\
Donec id elit non mi porta gravida at eget metus.
Fusce dapibus, tellus ac cursus commodo, tortor mauris condimentum
nibh, ut fermentum massa justo sit amet risus. Etiam porta sem
malesuada magna mollis euismod. Donec sed odio dui. Donec id elit non
mi porta gravida at eget metus. Fusce dapibus, tellus ac cursus
commodo, tortor mauris condimentum nibh, ut fermentum massa justo sit
amet risus. Etiam porta sem malesuada magna mollis euismod. Donec sed
odio dui."""
                        ),
                        dbc.Button("View details", color="secondary"),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.H2("Graph"),
                        dcc.Graph(
                            figure={"data": [{"x": [1, 2, 3], "y": [1, 4, 9]}]}
                        ),
                    ]
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        # dbc.Table(
                        #     table_header + table_body,
                        #     bordered=True,
                        #     size="lg",
                        #     # style=table_style,
                        #     style={"writingMode": "vertical-lr"},
                        #     responsive=True,
                        #     hover=True,
                        #     striped=True,
                        #     # className="vrt-header"
                        # )
                        dbc.Table.from_dataframe(
                            df,
                            striped=True,
                            bordered=True,
                            hover=True,
                            className="vrt-header",
                            style={"writingMode": "vertical-lr"},
                        )
                    ]
                )
            ]
        )
    ],
    className="mt-4",
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([navbar, body])

if __name__ == "__main__":
    app.run_server(debug=True)