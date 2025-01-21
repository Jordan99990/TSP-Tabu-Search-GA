import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
from tsp_solver import TabuSearchSolver, GeneticAlgorithmSolver, OptimalTSPSolver
import time
from ui import layout

app = dash.Dash(__name__)
app.title = "TSP Comparison: Tabu Search vs Genetic Algorithm"

np.random.seed(42)
num_cities = 45
cities = np.random.randint(0, 100, (num_cities, 2))

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
    if n_clicks == 0:
        return dash.no_update

    # Solve TSP with Tabu Search
    start_time = time.time()
    tabu_solver = TabuSearchSolver(cities, tabu_tenure, tabu_max_iterations, tabu_neighborhood_size, tabu_neighborhood_structure)
    tabu_path, tabu_distance, tabu_history = tabu_solver.solve()
    tabu_time = time.time() - start_time

    # Solve TSP with Genetic Algorithm
    start_time = time.time()
    ga_solver = GeneticAlgorithmSolver(cities, ga_population, ga_mutation_rate, ga_generations, ga_crossover_operator, ga_selection_operator, ga_mutation_operator)
    ga_path, ga_distance, ga_history = ga_solver.solve()
    ga_time = time.time() - start_time
    
    optimal_solver = OptimalTSPSolver(cities)
    optimal_path, optimal_distance = optimal_solver.solve()
    optimal_paths = [optimal_path]
    optimal_distances = [optimal_distance]

    avg_optimal_distance = np.mean(optimal_distances)
    best_optimal_path = optimal_paths[np.argmin(optimal_distances)]
    optimal_time = time.time() - start_time

    # Calculate statistics
    tabu_avg_improvement = round(np.mean(np.diff(tabu_history)), 2)
    ga_avg_improvement = round(np.mean(np.diff(ga_history)), 2)

    tabu_variance = round(np.var(tabu_history), 2)
    ga_variance = round(np.var(ga_history), 2)

    tabu_iterations_to_optimal = np.argmin(tabu_history)
    ga_iterations_to_optimal = np.argmin(ga_history)

    tabu_unique_solutions = len(set(map(tuple, tabu_solver.get_neighborhood(tabu_path))))
    ga_population_diversity = len(set(map(tuple, [ga_solver.create_random_solution() for _ in range(ga_population)])))

    # Plot all cities
    cities_fig = go.Figure()
    cities_fig.add_trace(go.Scatter(x=cities[:, 0], y=cities[:, 1], mode='markers', name="Cities"))
    cities_fig.update_layout(title="All TSP Cities", xaxis_title="X Coordinate", yaxis_title="Y Coordinate")

    # Plot results for Tabu Search
    tabu_result_fig = go.Figure()
    tabu_result_fig.add_trace(go.Scatter(x=cities[:, 0], y=cities[:, 1], mode='markers', name="Cities"))
    tabu_result_fig.add_trace(go.Scatter(x=cities[tabu_path + [tabu_path[0]], 0], y=cities[tabu_path + [tabu_path[0]], 1], mode='lines+markers', name="Tabu Search Route", line=dict(color='blue')))
    tabu_result_fig.update_layout(title="Tabu Search TSP Path", xaxis_title="X Coordinate", yaxis_title="Y Coordinate")
    
    # Plot results for Genetic Algorithm
    ga_result_fig = go.Figure()
    ga_result_fig.add_trace(go.Scatter(x=cities[:, 0], y=cities[:, 1], mode='markers', name="Cities"))
    ga_result_fig.add_trace(go.Scatter(x=cities[ga_path + [ga_path[0]], 0], y=cities[ga_path + [ga_path[0]], 1], mode='lines+markers', name="Genetic Algorithm Route", line=dict(color='red')))
    ga_result_fig.update_layout(title="Genetic Algorithm TSP Path", xaxis_title="X Coordinate", yaxis_title="Y Coordinate")

    # Plot results for Optimal TSP
    optimal_result_fig = go.Figure()
    optimal_result_fig.add_trace(go.Scatter(x=cities[:, 0], y=cities[:, 1], mode='markers', name="Cities"))
    optimal_result_fig.add_trace(go.Scatter(x=cities[best_optimal_path][:, 0], y=cities[best_optimal_path][:, 1], mode='lines+markers', name="Optimal TSP Route", line=dict(color='green')))
    optimal_result_fig.update_layout(title="Optimal TSP Path", xaxis_title="X Coordinate", yaxis_title="Y Coordinate")

    # Plot convergence
    convergence_fig = go.Figure()
    convergence_fig.add_trace(go.Scatter(y=tabu_history, mode='lines', name="Tabu Search Convergence"))
    convergence_fig.add_trace(go.Scatter(y=ga_history, mode='lines', name="Genetic Algorithm Convergence"))
    convergence_fig.update_layout(title="Current Iteration Convergence Comparison", xaxis_title="Iteration", yaxis_title="Distance")

    # Plot convergence over iterations
    iteration_convergence_fig = go.Figure()
    for i, tabu_data in enumerate(tabu_params_data):
        iteration_convergence_fig.add_trace(go.Scatter(y=tabu_data["history"], mode='lines', name=f"Tabu Search Iteration {i+1}"))
    for i, ga_data in enumerate(ga_params_data):
        iteration_convergence_fig.add_trace(go.Scatter(y=ga_data["history"], mode='lines', name=f"Genetic Algorithm Iteration {i+1}"))
    # Add current iteration
    iteration_convergence_fig.add_trace(go.Scatter(y=tabu_history, mode='lines', name=f"Tabu Search Iteration {n_clicks}"))
    iteration_convergence_fig.add_trace(go.Scatter(y=ga_history, mode='lines', name=f"Genetic Algorithm Iteration {n_clicks}"))
    iteration_convergence_fig.update_layout(title="Convergence Over Iterations", xaxis_title="Iteration", yaxis_title="Distance")

    results_table = html.Div([
        html.H3(f"Current Benchmark Iteration {n_clicks}", style={"textAlign": "center"}),
        html.Table([
            html.Thead([
                html.Tr([html.Th("Algorithm"), html.Th("Best Distance"), html.Th("Best Solution Cities")])
            ]),
            html.Tbody([
                html.Tr([html.Td("Tabu Search", style={"border": "1px solid black"}), html.Td(round(tabu_distance, 2), style={"border": "1px solid black"}), html.Td(str(cities[tabu_path + [tabu_path[0]]]), style={"border": "1px solid black"})]),
                html.Tr([html.Td("Genetic Algorithm", style={"border": "1px solid black"}), html.Td(round(ga_distance, 2), style={"border": "1px solid black"}), html.Td(str(cities[ga_path + [ga_path[0]]]), style={"border": "1px solid black"})]),
                html.Tr([html.Td("Optimal TSP", style={"border": "1px solid black"}), html.Td(round(avg_optimal_distance, 2), style={"border": "1px solid black"}), html.Td(str(cities[best_optimal_path]), style={"border": "1px solid black"})])
            ])
        ], style={"width": "100%", "border": "1px solid black", "border-collapse": "collapse", "margin-top": "20px", "text-align": "center"})
    ])

    tabu_params_data.append({"iteration": n_clicks, "tabu_tenure": tabu_tenure, "max_iterations": tabu_max_iterations, "neighborhood_size": tabu_neighborhood_size, "neighborhood_structure": tabu_neighborhood_structure, "best_distance": round(tabu_distance, 2), "time": tabu_time, "avg_improvement": tabu_avg_improvement, "variance": tabu_variance, "iterations_to_optimal": tabu_iterations_to_optimal, "unique_solutions": tabu_unique_solutions, "history": tabu_history})
    ga_params_data.append({"iteration": n_clicks, "population_size": ga_population, "mutation_rate": ga_mutation_rate, "generations": ga_generations, "crossover_operator": ga_crossover_operator, "selection_operator": ga_selection_operator, "mutation_operator": ga_mutation_operator, "best_distance": round(ga_distance, 2), "time": ga_time, "avg_improvement": ga_avg_improvement, "variance": ga_variance, "iterations_to_optimal": ga_iterations_to_optimal, "population_diversity": ga_population_diversity, "history": ga_history})

    # Distance Distribution Histogram
    distance_distribution_histogram = go.Figure()
    distance_distribution_histogram.add_trace(go.Histogram(x=[entry["best_distance"] for entry in tabu_params_data], name="Tabu Search"))
    distance_distribution_histogram.add_trace(go.Histogram(x=[entry["best_distance"] for entry in ga_params_data], name="Genetic Algorithm"))
    distance_distribution_histogram.update_layout(title="Distance Distribution Histogram", xaxis_title="Distance", yaxis_title="Count", barmode='overlay')
    distance_distribution_histogram.update_traces(opacity=0.75)

    # Box Plot of Distances
    distance_box_plot = go.Figure()
    distance_box_plot.add_trace(go.Box(y=[entry["best_distance"] for entry in tabu_params_data], name="Tabu Search"))
    distance_box_plot.add_trace(go.Box(y=[entry["best_distance"] for entry in ga_params_data], name="Genetic Algorithm"))
    distance_box_plot.update_layout(title="Box Plot of Distances", yaxis_title="Distance")

    # Scatter Plot of Execution Times
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

    # Heatmap of Distance Matrix
    distance_matrix = tabu_solver.create_distance_matrix()
    distance_matrix_heatmap = go.Figure(data=go.Heatmap(z=distance_matrix))
    distance_matrix_heatmap.update_layout(title="Heatmap of Distance Matrix", xaxis_title="City Index", yaxis_title="City Index")

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
                html.Tr([html.Th("Iteration"), html.Th("Population Size"), html.Th("Mutation Rate"), html.Th("Generations"), html.Th("Crossover Operator"), html.Th("Selection Operator"), html.Th("Mutation Operator"), html.Th("Best Distance"), html.Th("Time (s)"), html.Th("Avg Improvement"), html.Th("Variance"), html.Th("Iterations to Optimal"), html.Th("Population Diversity")])
            ]),
            html.Tbody([
                html.Tr([html.Td(row["iteration"], style={"border": "1px solid black"}), html.Td(row["population_size"], style={"border": "1px solid black"}), html.Td(row["mutation_rate"], style={"border": "1px solid black"}), html.Td(row["generations"], style={"border": "1px solid black"}), html.Td(row["crossover_operator"], style={"border": "1px solid black"}), html.Td(row["selection_operator"], style={"border": "1px solid black"}), html.Td(row["mutation_operator"], style={"border": "1px solid black"}), html.Td(row["best_distance"], style={"border": "1px solid black"}), html.Td(round(row["time"], 2), style={"border": "1px solid black"}), html.Td(row["avg_improvement"], style={"border": "1px solid black"}), html.Td(row["variance"], style={"border": "1px solid black"}), html.Td(row["iterations_to_optimal"], style={"border": "1px solid black"}), html.Td(row["population_diversity"], style={"border": "1px solid black"})]) for row in ga_params_data
            ])
        ], style={"width": "100%", "border": "1px solid black", "border-collapse": "collapse", "margin-top": "20px", "text-align": "center"})
    ])
    
    best_tabu = min(tabu_params_data, key=lambda x: x["best_distance"])
    best_ga = min(ga_params_data, key=lambda x: x["best_distance"])

    best_results_table = html.Div([
        html.H3("Best Results", style={"textAlign": "center"}),
        html.Table([
            html.Thead([
                html.Tr([html.Th("Algorithm"), html.Th("Best Iteration"), html.Th("Best Distance"), html.Th("Best Solution Cities"), html.Th("Time (s)")])
            ]),
            html.Tbody([
                html.Tr([html.Td("Tabu Search", style={"border": "1px solid black"}), html.Td(best_tabu["iteration"], style={"border": "1px solid black"}), html.Td(best_tabu["best_distance"], style={"border": "1px solid black"}), html.Td(str(cities[tabu_path + [tabu_path[0]]]), style={"border": "1px solid black"}), html.Td(round(best_tabu["time"], 2), style={"border": "1px solid black"})]),
                html.Tr([html.Td("Genetic Algorithm", style={"border": "1px solid black"}), html.Td(best_ga["iteration"], style={"border": "1px solid black"}), html.Td(best_ga["best_distance"], style={"border": "1px solid black"}), html.Td(str(cities[ga_path + [ga_path[0]]]), style={"border": "1px solid black"}), html.Td(round(best_ga["time"], 2), style={"border": "1px solid black"})])
            ])
        ], style={"width": "100%", "border": "1px solid black", "border-collapse": "collapse", "margin-top": "20px", "text-align": "center"})
    ])
    
    return (cities_fig, tabu_result_fig, ga_result_fig, optimal_result_fig, convergence_fig, results_table, 
            tabu_params_table, ga_params_table, tabu_params_data, ga_params_data, distance_distribution_histogram, 
            distance_box_plot, execution_time_scatter_plot, distance_matrix_heatmap, iteration_convergence_fig, best_results_table)

if __name__ == "__main__":
    app.run_server(debug=True)