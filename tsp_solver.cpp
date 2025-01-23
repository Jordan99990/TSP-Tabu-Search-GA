#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <nlohmann/json.hpp>
#include <omp.h>
#include <set>
#include <utility>
#include <numeric>

using namespace std;
using json = nlohmann::json;

struct City {
    int x, y;
};

void to_json(json& j, const City& city) {
    j = json{{"x", city.x}, {"y", city.y}};
}

struct TabuSearchResult {
    vector<int> best_solution;
    int best_distance;
    double avg_improvement;
    double variance;
    int iterations_to_optimal;
    int unique_solutions;
    vector<int> history; 
};

struct GeneticAlgorithmResult {
    vector<int> best_solution;
    int best_distance;
    double avg_improvement;
    double variance;
    int iterations_to_optimal;
    int unique_solutions;
    vector<int> history; 
};

vector<City> read_tsp(const string& file_path) {
    ifstream file(file_path);
    string line;
    vector<City> cities;
    bool node_coord_section = false;

    while (getline(file, line)) {
        if (line.find("NODE_COORD_SECTION") != string::npos) {
            node_coord_section = true;
            continue;
        }
        if (line.find("EOF") != string::npos) {
            break;
        }
        if (node_coord_section) {
            istringstream iss(line);
            int index;
            double x, y;
            iss >> index >> x >> y;
            cities.push_back({static_cast<int>(x), static_cast<int>(y)});
        }
    }
    return cities;
}

int calculate_distance(const City& a, const City& b) {
    return static_cast<int>(round(sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2))));
}

class OptimalTSPSolver {
public:
    OptimalTSPSolver(const vector<pair<double, double>>& cities) : cities(cities) {}

    pair<vector<int>, int> solve() {
        int num_cities = cities.size();
        auto distance_matrix = create_distance_matrix(num_cities);

        vector<vector<int>> mst = find_minimum_spanning_tree(distance_matrix, num_cities);
        vector<int> perfect_matching = find_perfect_matching(distance_matrix, mst, num_cities);
        vector<int> eulerian_path = find_eulerian_path(mst, perfect_matching, num_cities);
        vector<int> optimal_path = make_hamiltonian(eulerian_path);

        int optimal_distance = calculate_total_distance(optimal_path, distance_matrix);
        return {optimal_path, optimal_distance};
    }

private:
    vector<pair<double, double>> cities;

    vector<vector<double>> create_distance_matrix(int num_cities) {
        vector<vector<double>> distance_matrix(num_cities, vector<double>(num_cities, 0.0));
        for (int i = 0; i < num_cities; ++i) {
            for (int j = 0; j < num_cities; ++j) {
                distance_matrix[i][j] = hypot(cities[i].first - cities[j].first, cities[i].second - cities[j].second);
            }
        }
        return distance_matrix;
    }

    vector<vector<int>> find_minimum_spanning_tree(const vector<vector<double>>& distance_matrix, int num_cities) {
        vector<vector<int>> mst(num_cities);
        vector<bool> visited(num_cities, false);
        vector<double> min_distance(num_cities, numeric_limits<double>::infinity());
        vector<int> parent(num_cities, -1);

        min_distance[0] = 0;
        for (int i = 0; i < num_cities; ++i) {
            int u = -1;
            for (int v = 0; v < num_cities; ++v) {
                if (!visited[v] && (u == -1 || min_distance[v] < min_distance[u])) {
                    u = v;
                }
            }

            visited[u] = true;
            if (parent[u] != -1) {
                mst[u].push_back(parent[u]);
                mst[parent[u]].push_back(u);
            }

            for (int v = 0; v < num_cities; ++v) {
                if (!visited[v] && distance_matrix[u][v] < min_distance[v]) {
                    min_distance[v] = distance_matrix[u][v];
                    parent[v] = u;
                }
            }
        }
        return mst;
    }

    vector<int> find_perfect_matching(const vector<vector<double>>& distance_matrix, const vector<vector<int>>& mst, int num_cities) {
        vector<int> odd_degree_nodes;
        for (int i = 0; i < num_cities; ++i) {
            if (mst[i].size() % 2 == 1) {
                odd_degree_nodes.push_back(i);
            }
        }

        vector<bool> matched(odd_degree_nodes.size(), false);
        vector<int> matching;

        for (size_t i = 0; i < odd_degree_nodes.size(); ++i) {
            if (!matched[i]) {
                double min_distance = numeric_limits<double>::infinity();
                int best_match = -1;

                for (size_t j = i + 1; j < odd_degree_nodes.size(); ++j) {
                    if (!matched[j] && distance_matrix[odd_degree_nodes[i]][odd_degree_nodes[j]] < min_distance) {
                        min_distance = distance_matrix[odd_degree_nodes[i]][odd_degree_nodes[j]];
                        best_match = j;
                    }
                }

                if (best_match != -1) {
                    matched[i] = matched[best_match] = true;
                    matching.push_back(odd_degree_nodes[i]);
                    matching.push_back(odd_degree_nodes[best_match]);
                }
            }
        }

        return matching;
    }

    vector<int> find_eulerian_path(const vector<vector<int>>& mst, const vector<int>& perfect_matching, int num_cities) {
        vector<vector<int>> graph = mst;

        for (size_t i = 0; i < perfect_matching.size(); i += 2) {
            graph[perfect_matching[i]].push_back(perfect_matching[i + 1]);
            graph[perfect_matching[i + 1]].push_back(perfect_matching[i]);
        }

        vector<int> eulerian_path;
        vector<bool> visited(num_cities, false);
        dfs(0, graph, visited, eulerian_path);
        return eulerian_path;
    }

    void dfs(int node, const vector<vector<int>>& graph, vector<bool>& visited, vector<int>& path) {
        visited[node] = true;
        path.push_back(node);

        for (int neighbor : graph[node]) {
            if (!visited[neighbor]) {
                dfs(neighbor, graph, visited, path);
            }
        }
    }

    vector<int> make_hamiltonian(const vector<int>& eulerian_path) {
        vector<bool> visited(cities.size(), false);
        vector<int> hamiltonian_path;

        for (int node : eulerian_path) {
            if (!visited[node]) {
                visited[node] = true;
                hamiltonian_path.push_back(node);
            }
        }
        return hamiltonian_path;
    }

    int calculate_total_distance(const vector<int>& path, const vector<vector<double>>& distance_matrix) {
        int distance = 0;
        for (size_t i = 0; i < path.size(); ++i) {
            distance += static_cast<int>(distance_matrix[path[i]][path[(i + 1) % path.size()]]);
        }
        return distance;
    }
};

class TabuSearchSolver {
public:
    TabuSearchSolver(const vector<City>& cities, int tabu_tenure, int max_iterations, int neighborhood_size, const string& neighborhood_structure)
        : cities(cities), tabu_tenure(tabu_tenure), max_iterations(max_iterations), neighborhood_size(neighborhood_size), neighborhood_structure(neighborhood_structure) {}
    
    TabuSearchResult solve() {
        int num_cities = cities.size();
        std::vector<int> current_solution(num_cities);
        std::iota(current_solution.begin(), current_solution.end(), 0);
        std::shuffle(current_solution.begin(), current_solution.end(), rng);
        std::vector<int> best_solution = current_solution;
        int best_distance = calculate_total_distance(best_solution);
        std::vector<std::vector<int>> tabu_list;
        std::vector<int> tabu_history = {best_distance};

        int iterations_to_optimal = 0;
        double total_improvement = 0.0;

        for (int iter = 0; iter < max_iterations; ++iter) {
            std::vector<std::vector<int>> neighborhood = get_neighborhood(current_solution);
            neighborhood.erase(std::remove_if(neighborhood.begin(), neighborhood.end(), [&](const std::vector<int>& sol) {
                return std::find(tabu_list.begin(), tabu_list.end(), sol) != tabu_list.end();
            }), neighborhood.end());

            if (neighborhood.empty()) break;

            int min_distance = std::numeric_limits<int>::max();
            std::vector<int> best_neighbor;

            #pragma omp parallel for
            for (int i = 0; i < neighborhood.size(); ++i) {
                int current_distance = calculate_total_distance(neighborhood[i]);
                #pragma omp critical
                {
                    if (current_distance < min_distance) {
                        min_distance = current_distance;
                        best_neighbor = neighborhood[i];
                    }
                }
            }

            current_solution = best_neighbor;
            int current_distance = min_distance;

            if (current_distance < best_distance) {
                best_solution = current_solution;
                best_distance = current_distance;
                iterations_to_optimal = iter + 1;
            }

            tabu_list.push_back(current_solution);
            if (tabu_list.size() > tabu_tenure) {
                tabu_list.erase(tabu_list.begin());
            }

            total_improvement += (tabu_history.back() - best_distance);
            tabu_history.push_back(best_distance);
        }

        double avg_improvement = total_improvement / max_iterations;
        double variance = calculate_variance(tabu_history);
        int unique_solutions = calculate_unique_solutions(tabu_history);

        return {best_solution, best_distance, avg_improvement, variance, iterations_to_optimal, unique_solutions, tabu_history};
    }

private:
    vector<City> cities;
    int tabu_tenure;
    int max_iterations;
    int neighborhood_size;
    string neighborhood_structure;
    mt19937 rng{random_device{}()};

    int calculate_total_distance(const vector<int>& solution) {
        int distance = 0;
        for (size_t i = 0; i < solution.size(); ++i) {
            const City& city_a = cities[solution[i]];
            const City& city_b = cities[solution[(i + 1) % solution.size()]];
            distance += calculate_distance(city_a, city_b);
        }
        return distance;
    }

    vector<vector<int>> get_neighborhood(const vector<int>& solution) {
        if (neighborhood_structure == "2-opt") {
            return two_opt_neighborhood(solution);
        } else if (neighborhood_structure == "3-opt") {
            return three_opt_neighborhood(solution);
        } else if (neighborhood_structure == "shuffle") {
            return shuffle_subtour_neighborhood(solution);
        } else {
            throw invalid_argument("Invalid neighborhood structure");
        }
    }

    vector<vector<int>> two_opt_neighborhood(const vector<int>& solution) {
        vector<vector<int>> neighborhood;
        size_t n = solution.size();
        neighborhood.reserve(n * (n - 1) / 2); 

        #pragma omp parallel
        {
            vector<vector<int>> local_neighborhood;
            local_neighborhood.reserve(n * (n - 1) / 2 / omp_get_num_threads());

            #pragma omp for collapse(2) nowait
            for (size_t i = 0; i < n - 1; ++i) {
                for (size_t j = i + 1; j < n; ++j) {
                    vector<int> neighbor = solution;
                    reverse(neighbor.begin() + i, neighbor.begin() + j + 1);
                    local_neighborhood.push_back(move(neighbor)); 
                }
            }

            #pragma omp critical
            neighborhood.insert(neighborhood.end(), local_neighborhood.begin(), local_neighborhood.end());
        }

        return neighborhood;
    }

    vector<vector<int>> three_opt_neighborhood(const vector<int>& solution) {
        vector<vector<int>> neighborhood;
        size_t n = solution.size();
        neighborhood.reserve(n * (n - 1) * (n - 2) / 6);

        #pragma omp parallel
        {
            vector<vector<int>> local_neighborhood;
            local_neighborhood.reserve(n * (n - 1) * (n - 2) / 6 / omp_get_num_threads());

            #pragma omp for collapse(3) nowait
            for (size_t i = 0; i < n - 2; ++i) {
                for (size_t j = i + 1; j < n - 1; ++j) {
                    for (size_t k = j + 1; k < n; ++k) {
                        vector<int> neighbor = solution;
                        reverse(neighbor.begin() + i, neighbor.begin() + j + 1);
                        reverse(neighbor.begin() + j + 1, neighbor.begin() + k + 1);
                        local_neighborhood.push_back(move(neighbor)); 
                    }
                }
            }

            #pragma omp critical
            neighborhood.insert(neighborhood.end(), local_neighborhood.begin(), local_neighborhood.end());
        }

        return neighborhood;
    }

    vector<vector<int>> shuffle_subtour_neighborhood(const vector<int>& solution) {
        vector<vector<int>> neighborhood;
        size_t n = solution.size();
        neighborhood.reserve(n * (n - 1) / 2); 

        random_device rd;
        mt19937 g(rd());

        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            vector<int> neighbor = solution;
            shuffle(neighbor.begin(), neighbor.end(), g);
            #pragma omp critical
            neighborhood.push_back(move(neighbor)); 
        }
        return neighborhood;
    }

    double calculate_variance(const std::vector<int>& history) {
        double mean = std::accumulate(history.begin(), history.end(), 0.0) / history.size();
        double variance = 0.0;
        for (int value : history) {
            variance += std::pow(value - mean, 2);
        }
        return variance / history.size();
    }

    int calculate_unique_solutions(const std::vector<int>& history) {
        std::set<int> unique_solutions(history.begin(), history.end());
        return unique_solutions.size();
    }
};

class GeneticAlgorithmSolver {
public:
    GeneticAlgorithmSolver(const vector<City>& cities, int population_size, double mutation_rate, int generations, const string& crossover_operator, const string& selection_operator, const string& mutation_operator)
        : cities(cities), population_size(population_size), mutation_rate(mutation_rate), generations(generations), crossover_operator(crossover_operator), selection_operator(selection_operator), mutation_operator(mutation_operator) {}

    GeneticAlgorithmResult solve() {
        int num_cities = cities.size();
        vector<vector<int>> population(population_size);
        for (auto& solution : population) {
            solution = create_random_solution();
        }

        vector<int> best_solution = *std::min_element(population.begin(), population.end(), [&](const std::vector<int>& a, const std::vector<int>& b) {
            return calculate_total_distance(a) < calculate_total_distance(b);
        });
        int best_distance = calculate_total_distance(best_solution);
        vector<int> ga_history = {best_distance};

        double total_improvement = 0.0;
        int iterations_to_optimal = 0;

        for (int gen = 0; gen < generations; ++gen) {
            vector<vector<int>> new_population;

            while (new_population.size() < population_size) {
                auto [parent1, parent2] = select_parents(population);
                auto [child1, child2] = crossover(parent1, parent2);
                new_population.push_back(mutate(child1));
                new_population.push_back(mutate(child2));
            }

            population = new_population;

            vector<int> current_best_solution = *std::min_element(population.begin(), population.end(), [&](const std::vector<int>& a, const std::vector<int>& b) {
                return calculate_total_distance(a) < calculate_total_distance(b);
            });
            int current_best_distance = calculate_total_distance(current_best_solution);

            if (current_best_distance < best_distance) {
                best_solution = current_best_solution;
                best_distance = current_best_distance;
                iterations_to_optimal = gen + 1;
            }

            total_improvement += (ga_history.back() - best_distance);
            ga_history.push_back(best_distance);
        }

        double avg_improvement = total_improvement / generations;
        double variance = calculate_variance(ga_history);
        int unique_solutions = calculate_unique_solutions(ga_history);

        return {best_solution, best_distance, avg_improvement, variance, iterations_to_optimal, unique_solutions, ga_history};
    }

private:
    vector<City> cities;
    int population_size;
    double mutation_rate;
    int generations;
    string crossover_operator;
    string selection_operator;
    string mutation_operator;
    mt19937 rng{random_device{}()};

    int calculate_total_distance(const vector<int>& solution) {
        int distance = 0;
        for (size_t i = 0; i < solution.size(); ++i) {
            const City& city_a = cities[solution[i]];
            const City& city_b = cities[solution[(i + 1) % solution.size()]];
            distance += calculate_distance(city_a, city_b);
        }
        return distance;
    }

    vector<int> create_random_solution() {
        vector<int> solution(cities.size());
        iota(solution.begin(), solution.end(), 0);
        shuffle(solution.begin(), solution.end(), rng);
        return solution;
    }

    pair<vector<int>, vector<int>> select_parents(const vector<vector<int>>& population) {
        if (selection_operator == "tournament") {
            return tournament_selection(population);
        } else {
            uniform_int_distribution<int> dist(0, population.size() - 1);
            return {population[dist(rng)], population[dist(rng)]};
        }
    }

    pair<vector<int>, vector<int>> tournament_selection(const vector<vector<int>>& population) {
        int tournament_size = 5;
        vector<int> parent1 = *min_element(population.begin(), population.begin() + tournament_size, [&](const vector<int>& a, const vector<int>& b) {
            return calculate_total_distance(a) < calculate_total_distance(b);
        });
        vector<int> parent2 = *min_element(population.begin() + tournament_size, population.end(), [&](const vector<int>& a, const vector<int>& b) {
            return calculate_total_distance(a) < calculate_total_distance(b);
        });
        return {parent1, parent2};
    }

    pair<vector<int>, vector<int>> crossover(const vector<int>& parent1, const vector<int>& parent2) {
        if (crossover_operator == "pmx") {
            return pmx_crossover(parent1, parent2);
        } else {
            return ox_crossover(parent1, parent2);
        }
    }

    pair<vector<int>, vector<int>> pmx_crossover(const vector<int>& parent1, const vector<int>& parent2) {
        int size = parent1.size();
        vector<int> child1(size, -1), child2(size, -1);

        uniform_int_distribution<int> dist(0, size - 1);
        int start = dist(rng), end = dist(rng);
        if (start > end) swap(start, end);

        copy(parent1.begin() + start, parent1.begin() + end, child1.begin() + start);
        copy(parent2.begin() + start, parent2.begin() + end, child2.begin() + start);

        fill_child(child1, parent2, end, size);
        fill_child(child2, parent1, end, size);

        return {child1, child2};
    }

    pair<vector<int>, vector<int>> ox_crossover(const vector<int>& parent1, const vector<int>& parent2) {
        int size = parent1.size();
        vector<int> child1(size, -1), child2(size, -1);

        uniform_int_distribution<int> dist(0, size - 1);
        int start = dist(rng), end = dist(rng);
        if (start > end) swap(start, end);

        copy(parent1.begin() + start, parent1.begin() + end, child1.begin() + start);
        copy(parent2.begin() + start, parent2.begin() + end, child2.begin() + start);

        fill_child(child1, parent2, end, size);
        fill_child(child2, parent1, end, size);

        return {child1, child2};
    }

    void fill_child(vector<int>& child, const vector<int>& parent, int end, int size) {
        int current_pos = end;
        for (int gene : parent) {
            if (find(child.begin(), child.end(), gene) == child.end()) {
                if (current_pos >= size) current_pos = 0;
                child[current_pos++] = gene;
            }
        }
    }

    vector<int> mutate(vector<int> solution) {
        if (uniform_real_distribution<double>(0, 1)(rng) < mutation_rate) {
            if (mutation_operator == "swap") {
                return swap_mutation(solution);
            } else {
                return scramble_mutation(solution);
            }
        }
        return solution;
    }

    vector<int> swap_mutation(vector<int> solution) {
        uniform_int_distribution<int> dist(0, solution.size() - 1);
        int i = dist(rng), j = dist(rng);
        swap(solution[i], solution[j]);
        return solution;
    }

    vector<int> scramble_mutation(vector<int> solution) {
        uniform_int_distribution<int> dist(0, solution.size() - 1);
        int start = dist(rng), end = dist(rng);
        if (start > end) swap(start, end);
        shuffle(solution.begin() + start, solution.begin() + end, rng);
        return solution;
    }

    double calculate_variance(const std::vector<int>& history) {
        double mean = std::accumulate(history.begin(), history.end(), 0.0) / history.size();
        double variance = 0.0;
        for (int value : history) {
            variance += std::pow(value - mean, 2);
        }
        return variance / history.size();
    }

    int calculate_unique_solutions(const std::vector<int>& history) {
        std::set<int> unique_solutions(history.begin(), history.end());
        return unique_solutions.size();
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <tsp_file> <solver_type> [params...]" << endl;
        return 1;
    }

    string file_path = argv[1];
    string solver_type = argv[2];
    vector<City> cities = read_tsp(file_path);

    if (solver_type == "optimal") {
        vector<pair<double, double>> city_pairs;
        for (const auto& city : cities) {
            city_pairs.emplace_back(city.x, city.y);
        }

        OptimalTSPSolver optimal_solver(city_pairs);
        auto [optimal_solution, optimal_distance] = optimal_solver.solve();
        json output = {
            {"best_distance", optimal_distance},
            {"best_solution", optimal_solution},
            {"cities", cities}
        };
        cout << output.dump() << endl;
    } else if (solver_type == "tabu") {
        int tabu_tenure = stoi(argv[3]);
        int max_iterations = stoi(argv[4]);
        int neighborhood_size = stoi(argv[5]);
        string neighborhood_structure = argv[6];

        TabuSearchSolver tabu_solver(cities, tabu_tenure, max_iterations, neighborhood_size, neighborhood_structure);
        auto [tabu_solution, tabu_distance, tabu_avg_improvement, tabu_variance, tabu_iterations_to_optimal, tabu_unique_solutions, tabu_history] = tabu_solver.solve();
        json output = {
            {"best_distance", tabu_distance},
            {"best_solution", tabu_solution},
            {"avg_improvement", tabu_avg_improvement},
            {"variance", tabu_variance},
            {"iterations_to_optimal", tabu_iterations_to_optimal},
            {"unique_solutions", tabu_unique_solutions},
            {"history", tabu_history}
        };
        cout << output.dump() << endl;
    } else if (solver_type == "ga") {
        int population_size = stoi(argv[3]);
        double mutation_rate = stod(argv[4]);
        int generations = stoi(argv[5]);
        string crossover_operator = argv[6];
        string selection_operator = argv[7];
        string mutation_operator = argv[8];

        GeneticAlgorithmSolver ga_solver(cities, population_size, mutation_rate, generations, crossover_operator, selection_operator, mutation_operator);
        auto [ga_solution, ga_distance, ga_avg_improvement, ga_variance, ga_iterations_to_optimal, ga_unique_solutions, ga_history] = ga_solver.solve();
        json output = {
            {"best_distance", ga_distance},
            {"best_solution", ga_solution},
            {"avg_improvement", ga_avg_improvement},
            {"variance", ga_variance},
            {"iterations_to_optimal", ga_iterations_to_optimal},
            {"unique_solutions", ga_unique_solutions},
            {"history", ga_history}
        };
        cout << output.dump() << endl;
    } else {
        cerr << "Invalid solver type. Use 'tabu' or 'ga'." << endl;
        return 1;
    }

    return 0;
}