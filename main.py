# PSO adapted from: https://medium.com/analytics-vidhya/implementing-particle-swarm-optimization-pso-algorithm-in-python-9efc2eb179a6


import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import random
import heapq
import math
import json

dim = 2
target_israndom = False
map_filename = "map_50.txt"
robot_paths_filename = "robot_paths_50.json"
fig_foldername = "50x50_4"

# plt.rc('font', size=22)
# plt.rc('axes', titlesize=22)
# plt.rc('axes', labelsize=22)
# plt.rc('figure', titlesize=22)
# plt.rc('xtick', labelsize=18)
# plt.rc('ytick', labelsize=18)
# plt.rc('legend', fontsize=18)

class Robot:
    def __init__(self, max_velocity=5):
        self.search_space = Space()
        self.position = np.zeros(dim)#np.array([0, np.random.randint(0, self.search_space.gridsize-1)])#np.zeros(dim) #self.search_space.generate_item() #
        self.initial_position = self.position
        self.velocity = np.random.randint(0, max_velocity, dim)

        # particle best
        self.pbest_position = copy.deepcopy(self.position)
        self.pbest_value = float('inf')

        self.path = [[i] for i in self.position]

    def move(self, max_velocity=5):
        step = np.round(self.velocity)
        destination = self.position + step
        while True:
            # zero velocity
            if (step == 0).all():
                break
            else:
                # print("step: ", step)
                next_position = self.position + np.floor(step/max(abs(step)))

            # hit wall
            if (next_position >= self.search_space.gridsize * np.ones(dim)).any() or (
                        next_position < np.zeros(dim)).any():
                self.velocity = np.random.randint(-max_velocity, max_velocity, dim)
                #self.velocity = -step  # np.zeros(dim)
                # print("hit wall")
                break
            # hit obstacle
            elif self.search_space.map[int(next_position[1]), int(next_position[0])] == 1:
                # print("hit obstacle")
                # self.velocity = np.random.randint(-2, 2, dim)
                self.velocity = np.random.randint(-max_velocity, max_velocity, dim)

                break
            # reached target
            elif (next_position == self.search_space.target).all():
                self.position = next_position
                [self.path[i].append(self.position[i]) for i in range(len(self.path))]
                # print("reached target")
                break
            # reached destination
            elif (next_position == destination).all():
                self.position = next_position
                [self.path[i].append(self.position[i]) for i in range(len(self.path))]
                # print("reached point")
                break
            # step into new position
            else:
                self.position = next_position
                step = destination - self.position
                [self.path[i].append(self.position[i]) for i in range(len(self.path))]
                # print("continue")

class Space:
    def __init__(self):
        # load map
        self.map, self.obstacles, self.valid_gridpoints = self._load_map(map_filename)
        self.gridsize = self.map.shape[0]

        if target_israndom:
            self.target = self.generate_item()
        else:
            self.target = (self.gridsize - 1) * np.ones(dim)

        self.robots = []

        # global best
        self.gbest_position = np.random.uniform(0, self.gridsize, dim)
        self.gbest_value = float('inf')

        # saving figures
        self.pso_folder = "PSO_plots"
        if not os.path.exists(self.pso_folder):
            os.makedirs(self.pso_folder)
        self.pso_set_dir = os.path.join(self.pso_folder, fig_foldername)
        if not os.path.exists(self.pso_set_dir):
            os.makedirs(self.pso_set_dir)
        self.fignum = 0

    def _load_map(self, map_filename):
        #map_filename = "map2.txt"
        with open(map_filename, "r") as file:
            map_data = file.readlines()
        map_matrix = np.loadtxt(map_data)
        obstacles = []
        valid_gridpoints = []
        for y in range(len(map_matrix)):
            for x in range(len(map_matrix[y])):
                if map_matrix[y, x] == 1:
                    obstacles.append([x, y])
                elif map_matrix[y, x] == 0:
                    valid_gridpoints.append([x, y])

        return map_matrix, np.array(obstacles), valid_gridpoints

    def generate_item(self):
        while True:
            item = np.random.randint(self.map.shape[0], size=dim)
            if self.map[item[1], item[0]] == 0:
                break
        return item

    # lower fitness is better (randomness simulates signal noise)
    def pso_fitness(self, robot):
        return np.sum((self.target - robot.position)**2)**0.5 * (1 + random.uniform(-0.2, 0.2))

    def set_pbest(self):
        for robot in self.robots:
            fitness_candidate = self.pso_fitness(robot)
            if fitness_candidate < robot.pbest_value:
                robot.pbest_value = copy.deepcopy(fitness_candidate)
                robot.pbest_position = copy.deepcopy(robot.position)

    def set_gbest(self):
        for robot in self.robots:
            best_fitness_candidate = self.pso_fitness(robot)
            if best_fitness_candidate < self.gbest_value:
                self.gbest_value = copy.deepcopy(best_fitness_candidate)
                self.gbest_position = copy.deepcopy(robot.position)

    def move_robots(self, w, c1, c2, c3, max_velocity):
        for robot in self.robots:
            velocity_update = c1 * np.random.uniform(0, 1, dim) * (robot.pbest_position - robot.position) + \
                              c2 * np.random.uniform(0, 1, dim) * (self.gbest_position - robot.position) + \
                              np.mean([(c3 * np.random.uniform(0, 1, dim) * (neighbour.pbest_position - robot.position))
                                       for neighbour in self.robots])

            robot.velocity = w * (robot.velocity + velocity_update)
            # robot.velocity = w * (robot.velocity +
            # c1 * np.random.uniform(0, 1) * (robot.pbest_position - robot.position) +
            # c2 * np.random.uniform(0, 1) * (self.gbest_position - robot.position))
            robot.move(max_velocity)

    def shortest_path(self, target_error=3.0):
        path_lengths = []
        for robot in self.robots:
            dist = 0
            final_dist = np.sqrt(((np.array(robot.path[0][-1], robot.path[1][-1])-np.array(self.target))**2).sum())

            # robot reaches target
            if final_dist <= target_error:
                for i in range(len(robot.path[0]) - 1):
                    dist += np.sqrt(np.array([(robot.path[0][1] - robot.path[0][0])**2, (robot.path[1][1] - robot.path[1][0])**2]).sum())
                path_lengths.append(dist)
                # print("final dists: ", dist)
            else:
                path_lengths.append(500)
        # print("path lengths: ", path_lengths)
        shortest = min(path_lengths)
        shortest_path = self.robots[path_lengths.index(shortest)].path
        print('pso_path', shortest)
        return shortest, shortest_path

    def save_path(self, run_num=False):
        robot_paths = []
        for robot in self.robots:
            robot_paths.append(robot.path)
        # Writing JSON data
        if run_num:
            paths_filename = str(run_num) + robot_paths_filename
        else:
            paths_filename = robot_paths_filename
        with open(paths_filename, 'w') as f:
            json.dump(robot_paths, f)

    def visualise_pso(self, iteration, param_str):
        plt.figure(figsize=(5, 5))
        plt.scatter(self.obstacles[:,0], self.obstacles[:,1], marker='s', color="#000000")
        plt.scatter(self.target[0], self.target[1], marker='x', color="r")
        color_idx = 0
        for robot in self.robots:
            plt.plot(robot.position[0], robot.position[1], marker="v", color="C"+str(color_idx))
            plt.plot(robot.path[0], robot.path[1], color="C"+str(color_idx))
            color_idx += 1

        plt.text(x=-5 ,y=-1.2, s='Start')
        plt.text(x=self.target[0]-0.7, y=self.target[1]+2.5, s="Goal")
        # plt.text(x=6, y=-1.2, s='Iteration: %s' % (iteration))
        plt.text(x=20, y=-3.6, s='Iteration: %s' % (iteration))
        # plt.text(x=24, y=-1.2, s='No. of robots: %s' % (len(self.robots)))
        # plt.text(x=11, y=-5, s='$\chi$: %s , $C_1$: %s , $C_2$: %s , $C_3$: %s' % (w, c1, c2, c3))
        plt.text(x=3, y=-1.2, s=param_str)
        plt.xlim(-0.5, self.gridsize-0.5)
        plt.ylim(self.gridsize-0.5, -0.5)
        figname = "iter_" + str(iteration) + ".png"
        plt.savefig(os.path.join(self.pso_set_dir, figname))
        self.fignum += 1
        plt.show()

    def plot_shortest_path(self, shortest, shortest_path, iteration, param_str):
        plt.figure(figsize=(5, 5))
        plt.scatter(self.obstacles[:, 0], self.obstacles[:, 1], marker='s', color="#000000")
        plt.scatter(self.target[0], self.target[1], marker='x', color="r")
        color_idx = 0

        plt.text(x=-5, y=-1.2, s='Start')
        plt.text(x=self.target[0] - 0.7, y=self.target[1] + 2.5, s="Goal")
        plt.text(x=3, y=-3.6, s='No. of Iterations: %s, Shortest Path Length: %s' % (iteration, np.round(shortest,3)))
        # plt.text(x=6, y=-1.2, s='Shortest Path Length: %s' % np.round(shortest,3))
        plt.text(x=3, y=-1.2, s=param_str)

        # plt.text(x=24, y=-1.2, s='No. of robots: %s' % (len(self.robots)))

        plt.plot(shortest_path[0], shortest_path[1], "g-")
        plt.xlim(-0.5, self.gridsize - 0.5)
        plt.ylim(self.gridsize - 0.5, -0.5)
        plt.draw()
        figname = "pso_shortest.png"
        plt.savefig(os.path.join(self.pso_set_dir, figname))
        plt.show()


class GA:
    def __init__(self, use_pso, run_num=False):
        self.search_space = Space()
        self.valid_gridpoints = self.search_space.valid_gridpoints
        self.unexplored_grids = None
        self.population = []
        self.robot_paths = None

        if use_pso:
            self.generate_explored_map(viewing_range=7, run_num=run_num)

        self.valid_links = self._generate_valid_links()
        self.link_lengths = self._generate_link_distances()
        self.link_probabilities = self._generate_link_probabilities()
        self.gen_fitness = None
        self.elites = None
        self.best_fitness_history = []


        # saving figures
        self.ga_folder = "GA_plots"
        if not os.path.exists(self.ga_folder):
            os.makedirs(self.ga_folder)
        self.ga_set_dir = os.path.join(self.ga_folder, fig_foldername)
        if not os.path.exists(self.ga_set_dir):
            os.makedirs(self.ga_set_dir)

    def _generate_valid_links(self):
        # initialise array of list of valid links for each valid gripoint
        valid_links = -1 * np.ones((len(self.valid_gridpoints), 8))
        neighbours = [[-1, -1], [0, -1], [1, -1],
                      [-1, 0], [1, 0],
                      [-1, 1], [0, 1], [1, 1]]
        for gridpoint_idx in range(len(self.valid_gridpoints)):
            for neighbour_idx in range(len(neighbours)):
                neighbour = [(x + y) for x, y in zip(neighbours[neighbour_idx], self.valid_gridpoints[gridpoint_idx])]

                if neighbour in self.valid_gridpoints:
                    # set value of valid links to index in the list of valid gridpoints
                    valid_links[gridpoint_idx][neighbour_idx] = self.valid_gridpoints.index(neighbour)
            valid_links[gridpoint_idx][0] = gridpoint_idx

        return valid_links

    def _generate_link_distances(self):
        link_lengths = np.zeros((np.shape(self.valid_links)[0], np.shape(self.valid_links)[1] - 1))
        for i in range(np.shape(self.valid_links)[0]):
            for j in range(np.shape(self.valid_links)[1] - 1):
                pt1 = np.array(self.valid_gridpoints[int(self.valid_links[i][j])])
                pt2 = np.array(self.valid_gridpoints[int(self.valid_links[i][j + 1])])
                link_lengths[i][j] = self.distance(pt1, pt2)
        return link_lengths

    # generate link probabilities based on its length
    def _generate_link_probabilities(self):
        link_prob = np.zeros((np.shape(self.link_lengths)[0], np.shape(self.link_lengths)[1]))
        for i in range(np.shape(self.link_lengths)[0]):
            for j in range(np.shape(self.link_lengths)[1]):
                link_prob[i][j] = self.link_lengths[i][j] / (self.link_lengths[i]).sum()
        return link_prob

    # generates remaining gridpoints for a specified path to reach target
    def generate_path(self, path):
        current_gridpoint_idx = path[-1]
        while (self.valid_gridpoints[int(current_gridpoint_idx)] != self.search_space.target).any():
            free_path = self.valid_links[int(current_gridpoint_idx), 1:][self.valid_links[int(current_gridpoint_idx), 1:] > 0]
            # print(self.valid_gridpoints[int(current_gridpoint_idx)], free_path)
            current_gridpoint_idx = random.choice(free_path)
            path = np.append(path, current_gridpoint_idx)

        # todo: optimise complexity
        # # remove loops in path
        # unique_grid_ls = list(set(path))
        # seen_grid_idx = []
        # while len(unique_grid_ls) < len(path):
        #     for grid_idx in range(len(unique_grid_ls)):
        #         if grid_idx not in seen_grid_idx:
        #             seen_grid_idx.append(grid_idx)
        #             break
        #     grid_positions = [pos for pos, value in enumerate(path) if value == grid_idx]
        #     if len(grid_positions) > 1:
        #         path = np.append(path[:grid_positions[0]], path[grid_positions[-1]:])
        #         unique_grid_ls = list(set(path))
        return path

    def generate_explored_map(self, viewing_range, run_num):
        if run_num:
            paths_filename = str(run_num) + robot_paths_filename
        else:
            paths_filename = robot_paths_filename
        try:
            with open(paths_filename, 'r') as f:
                self.robot_paths = json.load(f)

        except Exception as e:
            print(e)
        if self.robot_paths is not None:
            explored_grids = []
            for path in self.robot_paths:
                for grid_idx in range(len(path[0])):
                    stepped_grid = [path[0][grid_idx], path[1][grid_idx]]
                    current_area = []
                    # wall, obstacle
                    for y in range(-viewing_range, viewing_range + 1):
                        for x in range(-viewing_range, viewing_range + 1):
                            grid = [stepped_grid[0] + x, stepped_grid[1] + y]
                            current_area += [[int(grid[0]), int(grid[1])]]
                            # # wall
                            # if (grid >= self.search_space.gridsize * np.ones(dim)).any() or (grid < np.zeros(dim)).any():
                            #     break
                            # # obstacle
                            # elif self.search_space.map[int(grid[1]), int(grid[0])] == 1:
                            #     break
                            # else:
                            #     current_area += [[int(grid[0]), int(grid[1])]]
                    for grid in current_area:
                        if grid not in explored_grids:
                            explored_grids += [grid]
            explored_grids += [[int(i) for i in self.search_space.target]]
            self.unexplored_grids = np.array([grid for grid in self.valid_gridpoints if grid not in explored_grids])
            # self.unexplored_grids = np.array(self.unexplored_grids)
            self.valid_gridpoints = [grid for grid in self.valid_gridpoints if grid in explored_grids]

            # plt.figure(figsize=(5,5))
            # plt.scatter(self.search_space.obstacles[:, 0], self.search_space.obstacles[:, 1], marker='s',
            #             color="#000000")
            # plt.scatter(self.search_space.target[0], self.search_space.target[1], marker='x', color="r")
            # if self.unexplored_grids is not None:
            #     plt.scatter(self.unexplored_grids[:, 0], self.unexplored_grids[:, 1], marker='s', color="#b8e4f2")
            #
            # color_idx = 0
            # for path in self.robot_paths:
            #     plt.plot(path[0][-1], path[1][-1], marker="v", color="C" + str(color_idx))
            #     plt.plot(path[0], path[1], color="C" + str(color_idx))
            #     color_idx += 1
            #
            # plt.xlim(-0.5, self.search_space.gridsize - 0.5)
            # plt.ylim(self.search_space.gridsize - 0.5, -0.5)
            # plt.draw()
            # plt.show()

    # convert pso robot paths into feasible completed GA path
    def convert_robot_path(self):
        if self.robot_paths is not None:
            for path_idx in range(len(self.robot_paths)):
                path = self.robot_paths[path_idx]
                self.population[path_idx] = np.array([])
                for i in range(len(path[0])):
                    self.population[path_idx] = np.append(self.population[path_idx],
                                                    self.valid_gridpoints.index([int(path[0][i]), int(path[1][i])]))
                self.population[path_idx] = self.generate_path(self.population[path_idx])


    def distance(self, pt1, pt2):
        return np.sqrt(((pt2 - pt1) ** 2).sum())

    # lower fitness is better
    def ga_fitness(self, path_idx):
        path_length = 0
        path = self.population[path_idx]
        for i in range(len(path) - 1):
            pt1 = np.array(self.valid_gridpoints[int(path[i])])
            pt2 = np.array(self.valid_gridpoints[int(path[i+1])])
            path_length += self.distance(pt1, pt2)
        return path_length

    def get_genbest(self, n_elites):
        self.gen_fitness = np.array([self.ga_fitness(path_idx) for path_idx in range(len(self.population))])
        self.elites = []
        elite_values = list(np.sort(list(set(heapq.nsmallest(n_elites, self.gen_fitness)))))
        # elite_chroms = np.array([np.where(gen_fitness == elite_value)[0] for elite_value in elite_values])
        for i in range(len(elite_values)):
            self.elites += list((np.where(self.gen_fitness == elite_values[i]))[0])
        self.best_fitness_history.append(self.gen_fitness[self.elites[0]])

    def generate_new_population(self, tournament_size, mutation_rate):
        new_population = []
        while len(new_population) < len(self.population):
            # child1 = Chromosome(generate_path = False)
            # child2 = Chromosome(generate_path = False)
            child1, child2 = self.crossover(tournament_size)
            new_population += [child1, child2]
        # replace with new generation and mutate
        self.mutate(new_population, mutation_rate)

    def crossover(self, tournament_size):
        iter = 0
        while iter < 3:
            parent1 = self.population[self.tournament_selection(tournament_size)]
            parent2 = self.population[self.tournament_selection(tournament_size)]

            # pick common gridpoint indices
            common_points = np.intersect1d(parent1, parent2)
            if len(common_points) > 0:
                crossover_point = random.choice(common_points)
                idx1 = np.where(parent1 == crossover_point)[0][0]
                idx2 = np.where(parent2 == crossover_point)[0][0]
                break
            # try again
            else:
                idx1 = 0
                idx2 = 0
            iter += 1

        # create children
        # if both crossover at start point or no common points, keep parents
        if idx1 == 0 and idx2 == 0:
            return parent1, parent2
        else:
            child1 = np.hstack((parent1[0:idx1], parent2[idx2:]))
            child2 = np.hstack((parent2[0:idx2], parent1[idx1:]))
            return child1, child2

    def tournament_selection(self, tournament_size):
        tournament = random.sample(range(len(self.population)), int(max(2, tournament_size)))
        # tournament = random.sample(range(len(self.population)), int(max(2,np.ceil(tournament_size*len(self.population)))))
        tournament_fitness = [self.gen_fitness[i] for i in tournament]
        best_chrom_idx = int(tournament[tournament_fitness.index(min(tournament_fitness))])
        return best_chrom_idx

    def mutate(self, population, mutation_rate):
        for path_idx in range(len(population)):
            if random.random() < mutation_rate:
                mutation_idx = random.choice(range(int(len(population[path_idx]))))
                # delete all nodes after mutation point
                population[path_idx] = population[path_idx][:mutation_idx + 1]
                # generate complete path from last node
                population[path_idx] = self.generate_path(population[path_idx])
        self.population = population

    def visualise_ga(self, generation, show_all_paths=False):
        plt.figure(figsize=(5, 5))
        plt.scatter(self.search_space.obstacles[:,0], self.search_space.obstacles[:,1], marker='s', color="#000000")
        plt.scatter(self.search_space.target[0], self.search_space.target[1], marker='x', color="r")
        if self.unexplored_grids is not None:
            plt.scatter(self.unexplored_grids[:, 0], self.unexplored_grids[:, 1], marker='s', color="#89cae0")
        color_idx = 0

        plt.text(x=-5, y=-1.2, s='Start')
        plt.text(x=self.search_space.target[0] - 0.7, y=self.search_space.target[1] + 2.5, s="Goal")
        plt.text(x=6, y=-1.2, s='Generation: %s' % (generation))
        plt.text(x=24, y=-1.2, s='Population Size: %s' % (len(self.population)))

        # plt.annotate('Start', xy=(self.valid_gridpoints[0][0]-0.5, self.valid_gridpoints[0][0]+2))
        # plt.annotate('Goal', xy=(self.search_space.target[0]-4, self.search_space.target[1]-0.5))
        # plt.text(x=-0.5, y=-1.5, s='Generation:(%s)' % (generation))

        # if len(self.elites) > 1:
        #     for path_idx in self.elites[1:]:
        #         path_x = [self.valid_gridpoints[int(point_idx)][0] for point_idx in self.population[path_idx]]
        #         path_y = [self.valid_gridpoints[int(point_idx)][1] for point_idx in self.population[path_idx]]
        #         plt.plot(path_x, path_y, "b--")

        if show_all_paths:
            for path in self.population:
                x = [self.valid_gridpoints[int(point_idx)][0] for point_idx in path]
                y = [self.valid_gridpoints[int(point_idx)][1] for point_idx in path]
                plt.plot(x, y)

        best_path_x = [self.valid_gridpoints[int(point_idx)][0] for point_idx in self.population[self.elites[0]]]
        best_path_y = [self.valid_gridpoints[int(point_idx)][1] for point_idx in self.population[self.elites[0]]]
        plt.plot(best_path_x, best_path_y, "g-")

        plt.xlim(-0.5, self.search_space.gridsize - 0.5)
        plt.ylim(self.search_space.gridsize - 0.5, -0.5)
        plt.draw()
        figname = "gen_" + str(generation) + ".png"
        plt.savefig(os.path.join(self.ga_set_dir, figname))
        plt.show()

def pso(n_robots=10, max_velocity=5, target_error=3.0, max_iterations=200,
        w=0.7298, c1=3., c2 = .8, c3 = 1.4,
        visualise=True, save_path=True, run_num=False):
    # Initialise PSO
    search_space = Space()
    robots_vector = [Robot(max_velocity) for _ in range(n_robots)]
    search_space.robots = robots_vector

    gbest_history = []
    iteration = 1
    while iteration < max_iterations:
        search_space.set_pbest()
        search_space.set_gbest()
        gbest_history.append(search_space.gbest_value)
        #print("Iter: ", iteration, "Best pos: ", search_space.gbest_position, "Best val: ", search_space.gbest_value)
        param_str = 'No. of robots: %s, $\chi$: %s, $C_1$: %s, $C_2$: %s, $C_3$: %s' % \
                    (len(search_space.robots), w, c1, c2, c3)
        if visualise:
            search_space.visualise_pso(iteration, param_str)

        if search_space.gbest_value < target_error:
            break

        search_space.move_robots(w, c1, c2, c3, max_velocity)
        # search_space.move_robots(w=0.7298, c1=3.6, c2=1., max_velocity=max_velocity)
        iteration += 1

    shortest, shortest_path = search_space.shortest_path(target_error)
    if visualise:
        search_space.plot_shortest_path(shortest, shortest_path, iteration, param_str)

    if save_path:
        search_space.save_path(run_num)

    # print("Target location: ", search_space.target)
    # print("PSO best solution is: ", search_space.gbest_position, "in n_iterations: ", iteration)
    # print("Target error: ", search_space.gbest_value)
    return iteration, gbest_history, shortest

def ga(population_size=100, n_elites=3, tournament_size=0.1, mutation_rate=0.1, max_generations=20,
       max_consecutive_generations=8, visualise=True, use_pso=True, run_num=False):
    # Initialise GA
    search_space = Space()
    path_finder = GA(use_pso, run_num)
    path_finder.population = [np.array([path_finder.valid_gridpoints.index([0, 0])])] * population_size

    if use_pso:
        path_finder.convert_robot_path()
        for path_idx in range(len(path_finder.robot_paths), population_size):
            path_finder.population[path_idx] = path_finder.generate_path(path_finder.population[path_idx])
    else:
        path_finder.population = [path_finder.generate_path(path) for path in path_finder.population]

    # GA loop
    generation = 1
    while generation < max_generations:
        path_finder.get_genbest(n_elites)
        #print(path_finder.best_fitness_history[-1])

        if visualise:
            path_finder.visualise_ga(generation, show_all_paths=False)
            # print(path_finder.best_fitness_history[-1])

        if generation > max_consecutive_generations:
            if path_finder.best_fitness_history[-1] == path_finder.best_fitness_history[-max_consecutive_generations]:
                print("no new best solution found")
                break
        path_finder.generate_new_population(tournament_size, mutation_rate)
        if generation == 1:
            print("generations: ", generation)
            print("distance of shortest path found: ", path_finder.best_fitness_history[-1])
        generation += 1
    print("generations: ", generation)
    print("distance of shortest path found: ", path_finder.best_fitness_history[-1])
    return generation, path_finder.best_fitness_history

def main():
    # map to use
    #map_filename = "map0.txt"

    # run once and visualise
    run_pso = True
    run_ga = True

    # change parameter values and plot graphs
    run_pso_param = False
    run_ga_param = False
    run_hybrid_param = False

    # PSO param search
    pso_param_name = "$C_2$"
    pso_param_range = np.linspace(0.05, 5, 50) #50
    pso_num_runs = 30 #30

    # GA param search
    ga_param_name = "Tournament Size"
    ga_param_range = np.linspace(2, 80, 15) #np.linspace(0.01, 0.6, 20)
    # pop_size = range(10, 30, 10)
    ga_num_runs = 10

    if run_pso:
        random.seed(1)
        iteration, gbest_history, shortest = pso(n_robots=10,
                                                 max_velocity=7,
                                                 target_error=2.0,
                                                 max_iterations=300,
                                                 w=0.7298, c1=3., c2=0.8, c3=0,
                                                 visualise=True,
                                                 save_path=True,
                                                 run_num = False)
        print(iteration, shortest)

    if run_hybrid_param:
        for i in range(ga_num_runs):
            random.seed(i)
            iteration, gbest_history, shortest = pso(n_robots=10,
                                                     max_velocity=7,
                                                     target_error=2.0,
                                                     max_iterations=300,
                                                     w=0.7298, c1=3., c2=0.8, c3=0,
                                                     visualise=False,
                                                     save_path=True,
                                                     run_num = i+1)
            print(iteration, gbest_history, shortest)

    if run_ga:
        generation, best_fitness_history = ga(population_size=200, #must be multiple of 2
                                              n_elites=2,
                                              tournament_size=6,
                                              mutation_rate=0.8,
                                              max_generations=100,
                                              max_consecutive_generations=10,
                                              visualise=True,
                                              use_pso=True,
                                              run_num=False)

    if run_pso_param:
        param_value_ls = []
        iter_ls = []
        iter_std_ls = []
        best_ls = []
        best_std_ls = []
        shortest_path_ls = []
        shortest_path_std_ls = []

        for param_value in pso_param_range:
            print(pso_param_name, ": ", param_value)
            param_iter_ls = []
            param_best_ls = []
            param_path_ls = []
            for i in range(pso_num_runs):
                # print("n_robots: ", n_robots, "run: ", i)
                random.seed(i)
                iteration, gbest_history, shortest = pso(n_robots=10,
                                                         max_velocity=7,
                                                         target_error=2.0,
                                                         max_iterations=300,
                                                         w=0.7298, c1=3, c2=param_value, c3=0,
                                                         # w=0.7298, c1=3., c2=.8, c3=1.4,
                                                         visualise=False,
                                                         save_path=False)
                param_iter_ls.append(iteration)
                param_best_ls.append(gbest_history[-1])
                param_path_ls.append(shortest)
            # save average and std to list
            param_value_ls.append(param_value)
            iter_ls.append(np.mean(param_iter_ls))
            iter_std_ls.append(np.std(param_iter_ls))
            best_ls.append(np.mean(param_best_ls))
            best_std_ls.append(np.std(param_best_ls))
            shortest_path_ls.append(np.mean(param_path_ls))
            shortest_path_std_ls.append(np.std(param_path_ls))

            print(iter_std_ls, best_std_ls, shortest_path_ls)
            # plot graphs
        plt.figure(0)
        plt.plot(param_value_ls, iter_ls, label="mean")
        plt.fill_between(param_value_ls, np.array(iter_ls) - np.array(iter_std_ls),
                         np.array(iter_ls) + np.array(iter_std_ls), alpha = 0.3, label="std")
        plt.xlabel(pso_param_name)
        plt.ylabel("No. of Iterations")
        plt.title('No. of Iterations vs ' + pso_param_name)
        plt.legend()
        plt.show()

        plt.figure(1)
        plt.plot(param_value_ls, shortest_path_ls, label="mean")
        plt.fill_between(param_value_ls, np.array(shortest_path_ls) - np.array(shortest_path_std_ls),
                         np.array(shortest_path_ls) + np.array(shortest_path_std_ls), alpha=0.3, label="std")
        plt.xlabel(pso_param_name)
        plt.ylabel("Shortest Path Length")
        plt.title('Shortest Path Length vs ' + pso_param_name)
        plt.legend()

        plt.draw()
        plt.show()


    if run_ga_param:
        param_value_ls = []
        n_gen_ls = []
        n_gen_std_ls = []
        best_dist_ls = []
        best_dist_std_ls = []

        for param_value in ga_param_range:
            param_gen_ls = []
            param_dist_ls = []
            for i in range(ga_num_runs):
                print(ga_param_name, ": ", param_value, "run: ", i)
                random.seed(i)
                generation, best_fitness_history = ga(population_size=200,
                                                      n_elites=2,
                                                      tournament_size=int(param_value), #0.2
                                                      mutation_rate=0.6, # 0.05
                                                      max_generations=100, #20
                                                      max_consecutive_generations=10, #10
                                                      visualise=False,
                                                      use_pso=True,
                                                      run_num=i+1)
                param_gen_ls.append(generation)
                param_dist_ls.append(best_fitness_history[-1])
            # save average and std to list
            param_value_ls.append(param_value)
            n_gen_ls.append(np.mean(param_gen_ls))
            n_gen_std_ls.append(np.std(param_gen_ls))
            best_dist_ls.append(np.mean(param_dist_ls))
            best_dist_std_ls.append(np.std(param_dist_ls))

            print(n_gen_std_ls, best_dist_std_ls)

            # todo: consider plotting "no. of iterations = gen * pop size"?

            # plot graphs
            plt.figure(0)
            plt.plot(param_value_ls, n_gen_ls, label="mean")
            plt.fill_between(param_value_ls, np.array(n_gen_ls) - np.array(n_gen_std_ls),
                             np.array(n_gen_ls) + np.array(n_gen_std_ls), alpha=0.3, label="std")
            plt.xlabel(ga_param_name)
            plt.ylabel("No. of Generations")
            plt.title('No. of Generations vs ' + ga_param_name)
            plt.show()

            plt.figure(1)
            plt.plot(param_value_ls, best_dist_ls, label="mean")
            plt.fill_between(param_value_ls, np.array(best_dist_ls) - np.array(best_dist_std_ls),
                             np.array(best_dist_ls) + np.array(best_dist_std_ls), alpha=0.3, label="std")
            plt.xlabel(ga_param_name)
            plt.ylabel("Shortest Path Length")
            plt.title('Shortest Path Length vs ' + ga_param_name)

            plt.draw()
            plt.show()


if __name__ == '__main__':
    main()