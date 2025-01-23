import subprocess
import json
import time
import concurrent.futures
import numpy as np
import random

class OptimalTSPSolver:
    def __init__(self, file_path):
        self.file_path = file_path

    def solve(self):
        result = subprocess.run(["./tsp_solver", self.file_path, "optimal"], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Solver stderr: {result.stderr}")
            raise ValueError(f"Solver error: {result.stderr}")

        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError:
            print(f"Solver stdout: {result.stdout}")
            raise ValueError(f"Failed to parse JSON output: {result.stdout}")

        optimal_solution = output["best_solution"]
        optimal_distance = output["best_distance"]
        cities = np.array([(city["x"], city["y"]) for city in output["cities"]])
        return optimal_solution, optimal_distance, cities

def run_cpp_solver(file_path, solver_type, params):
    command = ["./tsp_solver", file_path, solver_type] + [str(param) for param in params]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
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
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tabu_start_time = time.time()
        tabu_future = executor.submit(run_tabu_search, file_path, tabu_tenure, max_iterations, neighborhood_size, neighborhood_structure)
        
        ga_future = executor.submit(run_genetic_algorithm, file_path, population_size, mutation_rate, generations, crossover_operator, selection_operator, mutation_operator)
        
        tabu_result = tabu_future.result()
        tabu_time = time.time() - tabu_start_time
        
        ga_start_time = time.time()
        ga_result = ga_future.result()
        ga_time = time.time() - ga_start_time
    
    total_time = time.time() - start_time

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