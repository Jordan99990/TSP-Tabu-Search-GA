import numpy as np
import random
import networkx as nx
from networkx.algorithms.approximation import traveling_salesman_problem
from networkx.algorithms.approximation.traveling_salesman import christofides

class OptimalTSPSolver:
    def __init__(self, cities):
        self.cities = cities

    def solve(self):
        num_cities = len(self.cities)
        distance_matrix = self.create_distance_matrix()

        G = nx.Graph()
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                G.add_edge(i, j, weight=distance_matrix[i][j])

        optimal_path = traveling_salesman_problem(G, cycle=True, method=christofides)
        optimal_distance = self.calculate_total_distance(optimal_path)
        return optimal_path, optimal_distance

    def create_distance_matrix(self):
        num_cities = len(self.cities)
        distance_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                distance_matrix[i][j] = np.linalg.norm(self.cities[i] - self.cities[j])
        return distance_matrix

    def calculate_total_distance(self, path):
        distance = 0
        for i in range(len(path)):
            city_a = self.cities[path[i]]
            city_b = self.cities[path[(i + 1) % len(path)]]
            distance += np.linalg.norm(city_a - city_b)
        return distance
    
class TabuSearchSolver:
    def __init__(self, cities, tabu_tenure, max_iterations, neighborhood_size):
        self.cities = cities
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.neighborhood_size = neighborhood_size

    def solve(self):
        num_cities = len(self.cities)
        current_solution = list(range(num_cities))
        random.shuffle(current_solution)
        best_solution = current_solution[:]
        best_distance = self.calculate_total_distance(best_solution)
        tabu_list = []
        tabu_history = [best_distance]

        for _ in range(self.max_iterations):
            neighborhood = self.get_neighborhood(current_solution)
            neighborhood = [sol for sol in neighborhood if sol not in tabu_list]

            if not neighborhood:
                break

            current_solution = min(neighborhood, key=self.calculate_total_distance)
            current_distance = self.calculate_total_distance(current_solution)

            if current_distance < best_distance:
                best_solution = current_solution[:]
                best_distance = current_distance

            tabu_list.append(current_solution)
            if len(tabu_list) > self.tabu_tenure:
                tabu_list.pop(0)

            tabu_history.append(best_distance)

        return best_solution, best_distance, tabu_history

    def calculate_total_distance(self, solution):
        distance = 0
        for i in range(len(solution)):
            city_a = self.cities[solution[i]]
            city_b = self.cities[solution[(i + 1) % len(solution)]]
            distance += np.linalg.norm(city_a - city_b)
        return distance

    def get_neighborhood(self, solution):
        neighborhood = []
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                neighbor = solution[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighborhood.append(neighbor)
        return neighborhood

    def create_distance_matrix(self):
        num_cities = len(self.cities)
        distance_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                distance_matrix[i][j] = np.linalg.norm(self.cities[i] - self.cities[j])
        return distance_matrix

class GeneticAlgorithmSolver:
    def __init__(self, cities, population_size, mutation_rate, generations, crossover_operator, selection_operator, mutation_operator):
        self.cities = cities
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.crossover_operator = crossover_operator
        self.selection_operator = selection_operator
        self.mutation_operator = mutation_operator

    def solve(self):
        population = [self.create_random_solution() for _ in range(self.population_size)]
        best_solution = min(population, key=self.calculate_total_distance)
        best_distance = self.calculate_total_distance(best_solution)
        ga_history = [best_distance]

        for _ in range(self.generations):
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents(population)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([self.mutate(child1), self.mutate(child2)])

            population = new_population
            current_best_solution = min(population, key=self.calculate_total_distance)
            current_best_distance = self.calculate_total_distance(current_best_solution)

            if current_best_distance < best_distance:
                best_solution = current_best_solution
                best_distance = current_best_distance

            ga_history.append(best_distance)

        return best_solution, best_distance, ga_history

    def create_random_solution(self):
        solution = list(range(len(self.cities)))
        random.shuffle(solution)
        return solution

    def calculate_total_distance(self, solution):
        distance = 0
        for i in range(len(solution)):
            city_a = self.cities[solution[i]]
            city_b = self.cities[solution[(i + 1) % len(solution)]]
            distance += np.linalg.norm(city_a - city_b)
        return distance

    def select_parents(self, population):
        if self.selection_operator == "tournament":
            return self.tournament_selection(population)
        else:
            return random.sample(population, 2)

    def crossover(self, parent1, parent2):
        if self.crossover_operator == "pmx":
            return self.pmx_crossover(parent1, parent2)
        else:
            return self.ox_crossover(parent1, parent2)

    def mutate(self, solution):
        if random.random() < self.mutation_rate:
            if self.mutation_operator == "swap":
                return self.swap_mutation(solution)
            else:
                return self.scramble_mutation(solution)
        return solution

    def tournament_selection(self, population):
        tournament_size = 5
        tournament = random.sample(population, tournament_size)
        parent1 = min(tournament, key=self.calculate_total_distance)
        tournament = random.sample(population, tournament_size)
        parent2 = min(tournament, key=self.calculate_total_distance)
        return parent1, parent2

    def pmx_crossover(self, parent1, parent2):
        size = len(parent1)
        child1, child2 = [-1] * size, [-1] * size

        start, end = sorted(random.sample(range(size), 2))
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        fill_child(child1, parent2, end, size)
        fill_child(child2, parent1, end, size)

        return child1, child2

    def ox_crossover(self, parent1, parent2):
        size = len(parent1)
        child1, child2 = [-1] * size, [-1] * size

        start, end = sorted(random.sample(range(size), 2))
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        fill_child(child1, parent2, end, size)
        fill_child(child2, parent1, end, size)

        return child1, child2

    def swap_mutation(self, solution):
        i, j = random.sample(range(len(solution)), 2)
        solution[i], solution[j] = solution[j], solution[i]
        return solution

    def scramble_mutation(self, solution):
        start, end = sorted(random.sample(range(len(solution)), 2))
        subset = solution[start:end]
        random.shuffle(subset)
        solution[start:end] = subset
        return solution

def fill_child(child, parent, end, size):
    current_pos = end
    for gene in parent:
        if gene not in child:
            if current_pos >= size:
                current_pos = 0
            child[current_pos] = gene
            current_pos += 1