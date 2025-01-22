import subprocess
import json
import time
import concurrent.futures
from networkx.algorithms.approximation import traveling_salesman_problem
from networkx.algorithms.approximation.traveling_salesman import christofides
import numpy as np
import random
import networkx as nx

class OptimalTSPSolver:
    def __init__(self, cities):
        self.cities = cities

    def solve(self):
        num_cities = len(self.cities)
        distance_matrix = self.create_distance_matrix()

        G = nx.complete_graph(num_cities)
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                G[i][j]["weight"] = distance_matrix[i][j]

        optimal_path = traveling_salesman_problem(G, cycle=True, method=christofides)
        optimal_distance = self.calculate_total_distance(optimal_path, distance_matrix)
        return optimal_path, optimal_distance

    def create_distance_matrix(self):
        num_cities = len(self.cities)
        distance_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                distance_matrix[i][j] = np.linalg.norm(self.cities[i] - self.cities[j])
        return distance_matrix

    def calculate_total_distance(self, path, distance_matrix):
        distance = 0
        for i in range(len(path)):
            distance += distance_matrix[path[i]][path[(i + 1) % len(path)]]
        return int(distance)

def read_tsp(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    node_coord_section = False
    cities = []

    for line in lines:
        if line.startswith("NODE_COORD_SECTION"):
            node_coord_section = True
            continue
        if line.startswith("EOF"):
            break
        if node_coord_section:
            parts = line.split()
            if len(parts) == 3:
                cities.append((int(parts[1]), int(parts[2])))

    cities = np.array(cities)
    return cities

def run_cpp_solver(file_path, solver_type, params):
    command = ["./tsp_solver", file_path, solver_type] + [str(param) for param in params]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    # print("Command executed:", " ".join(command))  
    # print("Command output:", result.stdout)  
    # if result.stderr:
    #     print("Command error:", result.stderr)  
    
    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse JSON output: {result.stdout}")
    
    return output

def run_tabu_search(file_path, tabu_tenure, max_iterations, neighborhood_size, neighborhood_structure):
    params = [tabu_tenure, max_iterations, neighborhood_size, neighborhood_structure]
    output = run_cpp_solver(file_path, "tabu", params)
    return {
        "best_distance": output["best_distance"],
        "best_solution": output["best_solution"],
        "avg_improvement": output["avg_improvement"],
        "variance": output["variance"],
        "iterations_to_optimal": output["iterations_to_optimal"],
        "unique_solutions": output["unique_solutions"],
        "history": output["history"]
    }

def run_genetic_algorithm(file_path, population_size, mutation_rate, generations, crossover_operator, selection_operator, mutation_operator):
    params = [population_size, mutation_rate, generations, crossover_operator, selection_operator, mutation_operator]
    output = run_cpp_solver(file_path, "ga", params)
    return {
        "best_distance": output["best_distance"],
        "best_solution": output["best_solution"],
        "avg_improvement": output["avg_improvement"],
        "variance": output["variance"],
        "iterations_to_optimal": output["iterations_to_optimal"],
        "unique_solutions": output["unique_solutions"],
        "history": output["history"]
    }

def run_tsp_solvers(file_path, tabu_tenure=15, max_iterations=100, neighborhood_size=15, neighborhood_structure="2-opt", population_size=250, mutation_rate=0.05, generations=100, crossover_operator="pmx", selection_operator="tournament", mutation_operator="swap"):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        start_time = time.time()
        tabu_future = executor.submit(run_tabu_search, file_path, tabu_tenure, max_iterations, neighborhood_size, neighborhood_structure)
        ga_future = executor.submit(run_genetic_algorithm, file_path, population_size, mutation_rate, generations, crossover_operator, selection_operator, mutation_operator)
        
        tabu_start_time = time.time()
        tabu_result = tabu_future.result()
        tabu_end_time = time.time()
        
        ga_start_time = time.time()
        ga_result = ga_future.result()
        ga_end_time = time.time()
        
        total_time = time.time() - start_time

    tabu_time = tabu_end_time - tabu_start_time
    ga_time = ga_end_time - ga_start_time

    tabu_distance = tabu_result['best_distance']
    tabu_solution = tabu_result['best_solution']
    tabu_avg_improvement = tabu_result['avg_improvement']
    tabu_variance = tabu_result['variance']
    tabu_iterations_to_optimal = tabu_result['iterations_to_optimal']
    tabu_unique_solutions = tabu_result['unique_solutions']

    ga_distance = ga_result['best_distance']
    ga_solution = ga_result['best_solution']
    ga_avg_improvement = ga_result['avg_improvement']
    ga_variance = ga_result['variance']
    ga_iterations_to_optimal = ga_result['iterations_to_optimal']
    ga_unique_solutions = ga_result['unique_solutions']

    results = {
        "tabu_search": {
            "distance": tabu_distance,
            "best_solution": tabu_solution,
            "time": round(tabu_time, 2),
            "avg_improvement": round(tabu_avg_improvement, 2),
            "variance": round(tabu_variance, 2),
            "iterations_to_optimal": tabu_iterations_to_optimal,
            "unique_solutions": tabu_unique_solutions,
            "history": tabu_result['history']
        },
        "genetic_algorithm": {
            "distance": ga_distance,
            "best_solution": ga_solution,
            "time": round(ga_time, 2),
            "avg_improvement": round(ga_avg_improvement, 2),
            "variance": round(ga_variance, 2),
            "iterations_to_optimal": ga_iterations_to_optimal,
            "unique_solutions": ga_unique_solutions,
            "history": ga_result['history']
        }
    }
    
    print(results)
    
    return results