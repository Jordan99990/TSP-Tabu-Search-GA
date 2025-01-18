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
num_cities = 20
cities = np.random.rand(num_cities, 2) * 100

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
     Output("ga-params-store", "data")],
    [Input("run-button", "n_clicks")],
    [dash.dependencies.State("tabu-tenure", "value"),
     dash.dependencies.State("tabu-max-iterations", "value"),
     dash.dependencies.State("ga-population", "value"),
     dash.dependencies.State("ga-mutation-rate", "value"),
     dash.dependencies.State("ga-generations", "value"),
     dash.dependencies.State("tabu-params-store", "data"),
     dash.dependencies.State("ga-params-store", "data")]
)

def update_results(n_clicks, tabu_tenure, tabu_max_iterations, ga_population, ga_mutation_rate, ga_generations, tabu_params_data, ga_params_data):
    if n_clicks == 0:
        return dash.no_update

    # Solve TSP with Tabu Search
    start_time = time.time()
    tabu_solver = TabuSearchSolver(cities, tabu_tenure, tabu_max_iterations)
    tabu_path, tabu_distance, tabu_history = tabu_solver.solve()
    tabu_time = time.time() - start_time

    # Solve TSP with Genetic Algorithm
    start_time = time.time()
    ga_solver = GeneticAlgorithmSolver(cities, ga_population, ga_mutation_rate, ga_generations)
    ga_path, ga_distance, ga_history = ga_solver.solve()
    ga_time = time.time() - start_time

    # Solve TSP optimally
    start_time = time.time()
    optimal_solver = OptimalTSPSolver(cities)
    optimal_path, optimal_distance = optimal_solver.solve()
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
    tabu_result_fig.add_trace(go.Scatter(x=cities[tabu_path][:, 0], y=cities[tabu_path][:, 1], mode='lines+markers', name="Tabu Search Route", line=dict(color='blue')))
    tabu_result_fig.update_layout(title="Tabu Search TSP Path", xaxis_title="X Coordinate", yaxis_title="Y Coordinate")

    # Plot results for Genetic Algorithm
    ga_result_fig = go.Figure()
    ga_result_fig.add_trace(go.Scatter(x=cities[:, 0], y=cities[:, 1], mode='markers', name="Cities"))
    ga_result_fig.add_trace(go.Scatter(x=cities[ga_path][:, 0], y=cities[ga_path][:, 1], mode='lines+markers', name="Genetic Algorithm Route", line=dict(color='red')))
    ga_result_fig.update_layout(title="Genetic Algorithm TSP Path", xaxis_title="X Coordinate", yaxis_title="Y Coordinate")

    # Plot results for Optimal TSP
    optimal_result_fig = go.Figure()
    optimal_result_fig.add_trace(go.Scatter(x=cities[:, 0], y=cities[:, 1], mode='markers', name="Cities"))
    optimal_result_fig.add_trace(go.Scatter(x=cities[optimal_path][:, 0], y=cities[optimal_path][:, 1], mode='lines+markers', name="Optimal TSP Route", line=dict(color='green')))
    optimal_result_fig.update_layout(title="Optimal TSP Path", xaxis_title="X Coordinate", yaxis_title="Y Coordinate")

    # Plot convergence
    convergence_fig = go.Figure()
    convergence_fig.add_trace(go.Scatter(y=tabu_history, mode='lines', name="Tabu Search Convergence"))
    convergence_fig.add_trace(go.Scatter(y=ga_history, mode='lines', name="Genetic Algorithm Convergence"))
    convergence_fig.update_layout(title="Convergence Comparison", xaxis_title="Iteration", yaxis_title="Distance")

    # Generate results table
    results_table = html.Table([
        html.Thead([
            html.Tr([html.Th("Algorithm"), html.Th("Best Distance"), html.Th("Best Solution Cities")])
        ]),
        html.Tbody([
            html.Tr([html.Td("Tabu Search"), html.Td(round(tabu_distance, 2)), html.Td(str(cities[tabu_path]))]),
            html.Tr([html.Td("Genetic Algorithm"), html.Td(round(ga_distance, 2)), html.Td(str(cities[ga_path]))]),
            html.Tr([html.Td("Optimal TSP"), html.Td(round(optimal_distance, 2)), html.Td(str(cities[optimal_path]))])
        ])
    ], style={"width": "100%", "border": "1px solid black", "border-collapse": "collapse", "margin-top": "20px", "text-align": "center"}),

    tabu_params_data.append({"iteration": n_clicks, "tabu_tenure": tabu_tenure, "max_iterations": tabu_max_iterations, "best_distance": round(tabu_distance, 2), "time": tabu_time, "avg_improvement": tabu_avg_improvement, "variance": tabu_variance, "iterations_to_optimal": tabu_iterations_to_optimal, "unique_solutions": tabu_unique_solutions})
    ga_params_data.append({"iteration": n_clicks, "population_size": ga_population, "mutation_rate": ga_mutation_rate, "generations": ga_generations, "best_distance": round(ga_distance, 2), "time": ga_time, "avg_improvement": ga_avg_improvement, "variance": ga_variance, "iterations_to_optimal": ga_iterations_to_optimal, "population_diversity": ga_population_diversity})

    # Generate Tabu Search parameters table
    tabu_params_table = html.Table([
        html.Thead([
            html.Tr([html.Th("Iteration"), html.Th("Tabu Tenure"), html.Th("Max Iterations"), html.Th("Best Distance"), html.Th("Time (s)"), html.Th("Avg Improvement"), html.Th("Variance"), html.Th("Iterations to Optimal"), html.Th("Unique Solutions")])
        ]),
        html.Tbody([
            html.Tr([html.Td(row["iteration"]), html.Td(row["tabu_tenure"]), html.Td(row["max_iterations"]), html.Td(row["best_distance"]), html.Td(round(row["time"], 2)), html.Td(row["avg_improvement"]), html.Td(row["variance"]), html.Td(row["iterations_to_optimal"]), html.Td(row["unique_solutions"])]) for row in tabu_params_data
        ])
    ], style={"width": "100%", "border": "1px solid black", "border-collapse": "collapse", "margin-top": "20px", "text-align": "center"}),

    # Generate Genetic Algorithm parameters table
    ga_params_table = html.Table([
        html.Thead([
            html.Tr([html.Th("Iteration"), html.Th("Population Size"), html.Th("Mutation Rate"), html.Th("Generations"), html.Th("Best Distance"), html.Th("Time (s)"), html.Th("Avg Improvement"), html.Th("Variance"), html.Th("Iterations to Optimal"), html.Th("Population Diversity")])
        ]),
        html.Tbody([
            html.Tr([html.Td(row["iteration"]), html.Td(row["population_size"]), html.Td(row["mutation_rate"]), html.Td(row["generations"]), html.Td(row["best_distance"]), html.Td(round(row["time"], 2)), html.Td(row["avg_improvement"]), html.Td(row["variance"]), html.Td(row["iterations_to_optimal"]), html.Td(row["population_diversity"])]) for row in ga_params_data
        ])
    ], style={"width": "100%", "border": "1px solid black", "border-collapse": "collapse", "margin-top": "20px", "text-align": "center"})

    return cities_fig, tabu_result_fig, ga_result_fig, optimal_result_fig, convergence_fig, results_table, tabu_params_table, ga_params_table, tabu_params_data, ga_params_data

if __name__ == "__main__":
    app.run_server(debug=True)