from dash import dcc, html

layout = html.Div([
    html.H1("TSP Comparison: Tabu Search vs Genetic Algorithm", style={"textAlign": "center", "fontFamily": "Arial, sans-serif"}),

    html.Div([
        html.Label("Tabu Search Parameters", style={"fontWeight": "bold", "fontFamily": "Arial, sans-serif"}),
        html.Div([
            html.Label("Tabu Tenure", style={"fontFamily": "Arial, sans-serif"}),
            dcc.Input(id="tabu-tenure", type="number", value=10, step=1, style={
                'width': '100%',  
                'padding': '12px 20px',  
                'margin': '8px 0',  
                'boxSizing': 'border-box',  
                'border': '2px solid #ccc',  
                'borderRadius': '4px',  
                'fontSize': '16px',  
                'backgroundColor': '#f8f8f8',  
                'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)'  
            }),

            html.Label("Max Iterations", style={"fontFamily": "Arial, sans-serif"}),
            dcc.Input(id="tabu-max-iterations", type="number", value=100, step=1, style={
                'width': '100%',  
                'padding': '12px 20px',  
                'margin': '8px 0',  
                'boxSizing': 'border-box',  
                'border': '2px solid #ccc',  
                'borderRadius': '4px',  
                'fontSize': '16px',  
                'backgroundColor': '#f8f8f8',  
                'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)'  
            }),
        ], style={"margin-bottom": "20px"}),

        html.Label("Genetic Algorithm Parameters", style={"fontWeight": "bold", "fontFamily": "Arial, sans-serif"}),
        html.Div([
            html.Label("Population Size", style={"fontFamily": "Arial, sans-serif"}),
            dcc.Input(id="ga-population", type="number", value=50, step=1, style={
                'width': '100%',  
                'padding': '12px 20px',  
                'margin': '8px 0',  
                'boxSizing': 'border-box',  
                'border': '2px solid #ccc',  
                'borderRadius': '4px',  
                'fontSize': '16px',  
                'backgroundColor': '#f8f8f8',  
                'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)'  
            }),

            html.Label("Mutation Rate", style={"fontFamily": "Arial, sans-serif"}),
            dcc.Input(id="ga-mutation-rate", type="number", value=0.05, step=0.01, style={
                'width': '100%',  
                'padding': '12px 20px',  
                'margin': '8px 0',  
                'boxSizing': 'border-box',  
                'border': '2px solid #ccc',  
                'borderRadius': '4px',  
                'fontSize': '16px',  
                'backgroundColor': '#f8f8f8',  
                'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)'  
            }),

            html.Label("Generations", style={"fontFamily": "Arial, sans-serif"}),
            dcc.Input(id="ga-generations", type="number", value=100, step=1, style={
                'width': '100%',  
                'padding': '12px 20px',  
                'margin': '8px 0',  
                'boxSizing': 'border-box',  
                'border': '2px solid #ccc',  
                'borderRadius': '4px',  
                'fontSize': '16px',  
                'backgroundColor': '#f8f8f8',  
                'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)'  
            }),
        ], style={"margin-bottom": "20px"}),

        html.Button("Run Comparison", id="run-button", n_clicks=0, style={
            'backgroundColor': '#6348e8',  
            'color': 'white',  
            'padding': '15px 32px',  
            'textAlign': 'center',  
            'textDecoration': 'none',  
            'display': 'inline-block',  
            'fontSize': '16px',  
            'margin': '4px 2px',  
            'cursor': 'pointer',  
            'border': 'none',  
            'borderRadius': '12px'  
        }),
    ], style={
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center',
        'justifyContent': 'center',
        'width': '50%',
        'margin': '0 auto',
        'padding': '20px',
        'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)',
        'borderRadius': '10px',
        'backgroundColor': '#f9f9f9'
    }),

    html.Div([
        dcc.Graph(id="cities-graph", style={"display": "block", "margin": "0 auto", "width": "80%"}),
        dcc.Graph(id="tabu-result-graph", style={"display": "inline-block", "width": "30%"}),
        dcc.Graph(id="ga-result-graph", style={"display": "inline-block", "width": "30%"}),
        dcc.Graph(id="optimal-result-graph", style={"display": "inline-block", "width": "30%"}),
        dcc.Graph(id="convergence-graph", style={"display": "block", "margin": "0 auto", "width": "80%"}),
    ], style={"width": "100%", "textAlign": "center", "padding": "20px"}),

    html.Div(id="results-table", style={"margin-top": "20px"}),
    html.Div(id="tabu-params-table", style={"margin-top": "20px"}),
    html.Div(id="ga-params-table", style={"margin-top": "20px"}),
    
    dcc.Store(id="tabu-params-store", data=[]),
    dcc.Store(id="ga-params-store", data=[]),
])