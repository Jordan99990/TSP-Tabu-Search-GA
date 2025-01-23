import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
from tsp_solver_helper import OptimalTSPSolver, run_tsp_solvers
import time
from ui import layout

app = dash.Dash(__name__)
app.title = "TSP Comparison: Tabu Search vs Genetic Algorithm"

file_path = "u724.tsp"

optimal_solution = None
optimal_distance = None
cities = None

app.layout = layout

@app.callback(
    [Output("cities-graph", "figure"),
     Output("tabu-result-graph", "figure"),
     Output("ga-result-graph", "figure"),
     Output("optimal-result-graph", "figure"),
     Output("convergence-graph", "figure"),
     Output("results-table", "children"),
     Output("tabu-params-table", "children"),
     Output("ga-params-table", "children"), 
     Output("tabu-params-store", "data"),
     Output("ga-params-store", "data"),
     Output("distance-distribution-histogram", "figure"),
     Output("distance-box-plot", "figure"),
     Output("execution-time-scatter-plot", "figure"),
     Output("distance-matrix-heatmap", "figure"),
     Output("iteration-convergence-graph", "figure"),
     Output("best-results-table", "children")], 
    [Input("run-button", "n_clicks")],
    [dash.dependencies.State("tabu-tenure", "value"),
     dash.dependencies.State("tabu-max-iterations", "value"),
     dash.dependencies.State("tabu-neighborhood-size", "value"),
     dash.dependencies.State("tabu-neighborhood-structure", "value"),
     dash.dependencies.State("ga-population", "value"),
     dash.dependencies.State("ga-mutation-rate", "value"),
     dash.dependencies.State("ga-generations", "value"),
     dash.dependencies.State("ga-crossover-operator", "value"),
     dash.dependencies.State("ga-selection-operator", "value"),
     dash.dependencies.State("ga-mutation-operator", "value"),
     dash.dependencies.State("tabu-params-store", "data"),
     dash.dependencies.State("ga-params-store", "data")]
)

def update_results(n_clicks, tabu_tenure, tabu_max_iterations, tabu_neighborhood_size, tabu_neighborhood_structure, ga_population, ga_mutation_rate, ga_generations, ga_crossover_operator, ga_selection_operator, ga_mutation_operator, tabu_params_data, ga_params_data):
    global optimal_solution, optimal_distance, cities
    if n_clicks == 0:
        return dash.no_update

    results = run_tsp_solvers(file_path, tabu_tenure, tabu_max_iterations, tabu_neighborhood_size, tabu_neighborhood_structure, ga_population, ga_mutation_rate, ga_generations, ga_crossover_operator, ga_selection_operator, ga_mutation_operator)
    
    tabu_result = results["tabu_search"]
    ga_result = results["genetic_algorithm"]
    
    if optimal_solution is None or optimal_distance is None:
        optimal_solver = OptimalTSPSolver(file_path)
        optimal_solution, optimal_distance, cities = optimal_solver.solve()
    
    cities_fig = go.Figure()
    cities_fig.add_trace(go.Scatter(x=cities[:, 0], y=cities[:, 1], mode='markers', name="Cities"))
    cities_fig.update_layout(title="All TSP Cities", xaxis_title="X Coordinate", yaxis_title="Y Coordinate")

    tabu_params_data.append({
        "iteration": n_clicks,
        "tabu_tenure": tabu_tenure,
        "max_iterations": tabu_max_iterations,
        "neighborhood_size": tabu_neighborhood_size,
        "neighborhood_structure": tabu_neighborhood_structure,
        "best_distance": round(tabu_result["distance"], 2),
        "time": tabu_result["time"],
        "avg_improvement": tabu_result["avg_improvement"],
        "variance": tabu_result["variance"],
        "iterations_to_optimal": tabu_result["iterations_to_optimal"],
        "unique_solutions": tabu_result["unique_solutions"],
        "history": tabu_result["history"],
        "best_solution": tabu_result["best_solution"]
    })

    ga_params_data.append({
        "iteration": n_clicks,
        "population_size": ga_population,
        "mutation_rate": ga_mutation_rate,
        "generations": ga_generations,
        "crossover_operator": ga_crossover_operator,
        "selection_operator": ga_selection_operator,
        "mutation_operator": ga_mutation_operator,
        "best_distance": round(ga_result["distance"], 2),
        "time": ga_result["time"],
        "avg_improvement": ga_result["avg_improvement"],
        "variance": ga_result["variance"],
        "iterations_to_optimal": ga_result["iterations_to_optimal"],
        "unique_solutions": ga_result["unique_solutions"],
        "history": ga_result["history"],
        "best_solution": ga_result["best_solution"]
    })
    
    best_tabu = min(tabu_params_data, key=lambda x: x["best_distance"])
    best_ga = min(ga_params_data, key=lambda x: x["best_distance"])

    tabu_result_fig = go.Figure()
    tabu_result_fig.add_trace(go.Scatter(x=cities[:, 0], y=cities[:, 1], mode='markers', name="Cities"))
    tabu_result_fig.add_trace(go.Scatter(x=cities[best_tabu['best_solution'] + [best_tabu['best_solution'][0]], 0], y=cities[best_tabu['best_solution'] + [best_tabu['best_solution'][0]], 1], mode='lines+markers', name="Tabu Search Route", line=dict(color='blue')))
    tabu_result_fig.update_layout(title=f"Tabu Search TSP Path (<b>Best Distance: {round(best_tabu['best_distance'], 2)}, Iteration: {best_tabu['iteration']}</b>)", xaxis_title="X Coordinate", yaxis_title="Y Coordinate")

    ga_result_fig = go.Figure()
    ga_result_fig.add_trace(go.Scatter(x=cities[:, 0], y=cities[:, 1], mode='markers', name="Cities"))
    ga_result_fig.add_trace(go.Scatter(x=cities[best_ga['best_solution'] + [best_ga['best_solution'][0]], 0], y=cities[best_ga['best_solution'] + [best_ga['best_solution'][0]], 1], mode='lines+markers', name="Genetic Algorithm Route", line=dict(color='red')))
    ga_result_fig.update_layout(title=f"Genetic Algorithm TSP Path (<b>Best Distance: {round(best_ga['best_distance'], 2)}, Iteration: {best_ga['iteration']}</b>)", xaxis_title="X Coordinate", yaxis_title="Y Coordinate")

    optimal_result_fig = go.Figure()
    optimal_result_fig.add_trace(go.Scatter(x=cities[:, 0], y=cities[:, 1], mode='markers', name="Cities"))
    optimal_result_fig.add_trace(go.Scatter(x=cities[optimal_solution + [optimal_solution[0]], 0], y=cities[optimal_solution + [optimal_solution[0]], 1], mode='lines+markers', name="Optimal TSP Route", line=dict(color='green')))
    optimal_result_fig.update_layout(title=f"Optimal TSP Path (<b>Best Distance: {round(optimal_distance, 2)}</b>)", xaxis_title="X Coordinate", yaxis_title="Y Coordinate")

    convergence_fig = go.Figure()
    convergence_fig.add_trace(go.Scatter(y=tabu_result["history"], mode='lines', name="Tabu Search Convergence"))
    convergence_fig.add_trace(go.Scatter(y=ga_result["history"], mode='lines', name="Genetic Algorithm Convergence"))
    convergence_fig.update_layout(title="Current Iteration Convergence Comparison", xaxis_title="Iteration", yaxis_title="Distance")

    iteration_convergence_fig = go.Figure()
    for i, tabu_data in enumerate(tabu_params_data):
        iteration_convergence_fig.add_trace(go.Scatter(y=tabu_data["history"], mode='lines', name=f"Tabu Search Iteration {i+1}"))
    for i, ga_data in enumerate(ga_params_data):
        iteration_convergence_fig.add_trace(go.Scatter(y=ga_data["history"], mode='lines', name=f"Genetic Algorithm Iteration {i+1}"))

    execution_time_scatter_plot = go.Figure()
    execution_time_scatter_plot.add_trace(go.Scatter(x=[entry["iteration"] for entry in tabu_params_data], y=[entry["time"] for entry in tabu_params_data], mode='markers', name="Tabu Search"))
    execution_time_scatter_plot.add_trace(go.Scatter(x=[entry["iteration"] for entry in ga_params_data], y=[entry["time"] for entry in ga_params_data], mode='markers', name="Genetic Algorithm"))
    execution_time_scatter_plot.update_layout(
        title="Scatter Plot of Execution Times",
        xaxis_title="Iteration",
        yaxis_title="Time (s)",
        xaxis=dict(
            tickmode='linear',
            dtick=1
        )
    )

    distance_matrix = np.zeros((len(cities), len(cities)))
    for i in range(len(cities)):
        for j in range(len(cities)):
            distance_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])
    distance_matrix_heatmap = go.Figure(data=go.Heatmap(z=distance_matrix))
    distance_matrix_heatmap.update_layout(title="Heatmap of Distance Matrix", xaxis_title="City Index", yaxis_title="City Index")

    distance_distribution_histogram = go.Figure()
    distance_distribution_histogram.add_trace(go.Histogram(x=[entry["best_distance"] for entry in tabu_params_data], name="Tabu Search"))
    distance_distribution_histogram.add_trace(go.Histogram(x=[entry["best_distance"] for entry in ga_params_data], name="Genetic Algorithm"))
    distance_distribution_histogram.update_layout(
        title="Distance Distribution Histogram",
        xaxis_title="Distance",
        yaxis_title="Count",
        barmode='overlay'
    )
    distance_distribution_histogram.update_traces(opacity=0.75)

    distance_box_plot = go.Figure()
    distance_box_plot.add_trace(go.Box(y=[entry["best_distance"] for entry in tabu_params_data], name="Tabu Search"))
    distance_box_plot.add_trace(go.Box(y=[entry["best_distance"] for entry in ga_params_data], name="Genetic Algorithm"))
    distance_box_plot.update_layout(title="Box Plot of Best Distances", yaxis_title="Distance")

    results_table = html.Div([
        html.H3(f"Current Benchmark Iteration {n_clicks}", style={"textAlign": "center"}),
        html.Table([
            html.Thead([
                html.Tr([html.Th("Algorithm"), html.Th("Best Distance"), html.Th("Best Solution Cities")])
            ]),
            html.Tbody([
                html.Tr([html.Td("Tabu Search", style={"border": "1px solid black"}), html.Td(round(tabu_result["distance"], 2), style={"border": "1px solid black"}), html.Td(str(cities[tabu_result["best_solution"] + [tabu_result["best_solution"][0]]].tolist()), style={"border": "1px solid black"})]),
                html.Tr([html.Td("Genetic Algorithm", style={"border": "1px solid black"}), html.Td(round(ga_result["distance"], 2), style={"border": "1px solid black"}), html.Td(str(cities[ga_result["best_solution"] + [ga_result["best_solution"][0]]].tolist()), style={"border": "1px solid black"})])
            ])
        ], style={"width": "100%", "border": "1px solid black", "border-collapse": "collapse", "margin-top": "20px", "text-align": "center"})
    ])

    tabu_params_table = html.Div([
        html.H3("TSP with Tabu Search", style={"textAlign": "center"}),
        html.Table([
            html.Thead([
                html.Tr([html.Th("Iteration"), html.Th("Tabu Tenure"), html.Th("Max Iterations"), html.Th("Neighborhood Size"), html.Th("Neighborhood Structure"), html.Th("Best Distance"), html.Th("Time (s)"), html.Th("Avg Improvement"), html.Th("Variance"), html.Th("Iterations to Optimal"), html.Th("Unique Solutions")])
            ]),
            html.Tbody([
                html.Tr([html.Td(row["iteration"], style={"border": "1px solid black"}), html.Td(row["tabu_tenure"], style={"border": "1px solid black"}), html.Td(row["max_iterations"], style={"border": "1px solid black"}), html.Td(row["neighborhood_size"], style={"border": "1px solid black"}), html.Td(row["neighborhood_structure"], style={"border": "1px solid black"}), html.Td(row["best_distance"], style={"border": "1px solid black"}), html.Td(round(row["time"], 2), style={"border": "1px solid black"}), html.Td(row["avg_improvement"], style={"border": "1px solid black"}), html.Td(row["variance"], style={"border": "1px solid black"}), html.Td(row["iterations_to_optimal"], style={"border": "1px solid black"}), html.Td(row["unique_solutions"], style={"border": "1px solid black"})]) for row in tabu_params_data
            ])
        ], style={"width": "100%", "border": "1px solid black", "border-collapse": "collapse", "margin-top": "20px", "text-align": "center"})
    ])

    ga_params_table = html.Div([
        html.H3("TSP with Genetic Algorithm", style={"textAlign": "center"}),
        html.Table([
            html.Thead([
                html.Tr([html.Th("Iteration"), html.Th("Population Size"), html.Th("Mutation Rate"), html.Th("Generations"), html.Th("Crossover Operator"), html.Th("Selection Operator"), html.Th("Mutation Operator"), html.Th("Best Distance"), html.Th("Time (s)"), html.Th("Avg Improvement"), html.Th("Variance"), html.Th("Iterations to Optimal"), html.Th("Unique Solutions")])
            ]),
            html.Tbody([
                html.Tr([html.Td(row["iteration"], style={"border": "1px solid black"}), html.Td(row["population_size"], style={"border": "1px solid black"}), html.Td(row["mutation_rate"], style={"border": "1px solid black"}), html.Td(row["generations"], style={"border": "1px solid black"}), html.Td(row["crossover_operator"], style={"border": "1px solid black"}), html.Td(row["selection_operator"], style={"border": "1px solid black"}), html.Td(row["mutation_operator"], style={"border": "1px solid black"}), html.Td(row["best_distance"], style={"border": "1px solid black"}), html.Td(round(row["time"], 2), style={"border": "1px solid black"}), html.Td(row["avg_improvement"], style={"border": "1px solid black"}), html.Td(row["variance"], style={"border": "1px solid black"}), html.Td(row["iterations_to_optimal"], style={"border": "1px solid black"}), html.Td(row["unique_solutions"], style={"border": "1px solid black"})]) for row in ga_params_data
            ])
        ], style={"width": "100%", "border": "1px solid black", "border-collapse": "collapse", "margin-top": "20px", "text-align": "center"})
    ])

    return (cities_fig, tabu_result_fig, ga_result_fig, optimal_result_fig, convergence_fig, results_table, 
            tabu_params_table, ga_params_table, tabu_params_data, ga_params_data, distance_distribution_histogram, 
            distance_box_plot, execution_time_scatter_plot, distance_matrix_heatmap, iteration_convergence_fig, results_table)

if __name__ == "__main__":
    app.run_server(debug=True)