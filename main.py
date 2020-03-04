# PSO adapted from: https://medium.com/analytics-vidhya/implementing-particle-swarm-optimization-pso-algorithm-in-python-9efc2eb179a6

import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import random
import heapq
import math

dim = 2
target_israndom = False
map_filename = "map0.txt"

class Robot:
    def __init__(self):
        self.search_space = Space()
        self.position = np.zeros(dim)#np.array([0, np.random.randint(0, self.search_space.gridsize-1)])#np.zeros(dim) #self.search_space.generate_item() #
        self.initial_position = self.position
        self.velocity = np.random.randint(0, 7, dim)

        # particle best
        self.pbest_position = copy.deepcopy(self.position)
        self.pbest_value = float('inf')

        self.path = [[i] for i in self.position]

    def move(self):
        step = np.round(self.velocity)
        destination = self.position + step
        while True:
            # zero velocity
            if (step == 0).all():
                break
            else:
                # print("step: ", step)
                next_position = self.position + np.round(step/max(abs(step)))

            # hit wall
            if (next_position >= self.search_space.gridsize * np.ones(dim)).any() or (
                        next_position < np.zeros(dim)).any():
                self.velocity = -step  # np.zeros(dim)
                # print("hit wall")
                break
            # hit obstacle
            elif self.search_space.map[int(next_position[1]), int(next_position[0])] == 1:
                # print("hit obstacle")
                self.velocity = np.random.randint(-7, 7, dim)
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
        self.pso_set_dir = os.path.join(self.pso_folder, "set_1")
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

    # lower fitness is better
    def pso_fitness(self, robot):
        return np.sum((self.target - robot.position)**2)**0.5

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

    def move_robots(self, w, c1, c2):
        for robot in self.robots:
            robot.velocity = w * (robot.velocity +
                           c1 * np.random.uniform(0, 1) * (robot.pbest_position - robot.position) +
                           c2 * np.random.uniform(0, 1) * (self.gbest_position - robot.position))
            robot.move()


    def visualise_pso(self):
        #plt.ion()
        plt.figure(figsize=(5, 5))
        plt.scatter(self.obstacles[:,0], self.obstacles[:,1], marker='s', color="#000000")
        plt.scatter(self.target[0], self.target[1], marker='x', color="C1")
        color_idx = 0
        for robot in self.robots:
            #plt.plot(robot.initial_position[0], robot.initial_position[1], marker="o")
            plt.plot(robot.position[0], robot.position[1], marker="v", color="C"+str(color_idx))
            plt.plot(robot.path[0], robot.path[1], color="C"+str(color_idx))
            color_idx += 1
        plt.xlim(-0.5, self.gridsize-0.5)
        plt.ylim(self.gridsize-0.5, -0.5)
        figname = "iter_" + str(self.fignum) + ".png"
        plt.savefig(os.path.join(self.pso_set_dir, figname))
        self.fignum += 1
        plt.show()

class Chromosome:
    def __init__(self, generate_path=True):
        self.search_space = Space()
        self.path_finder = GA()
        if generate_path:
            self.path = np.array([self.path_finder.valid_gridpoints.index([0, 0])])
            self.generate_path()
        else:
            self.path = np.array([])

    def generate_path(self):
        current_gridpoint_idx = self.path[-1]

        while (self.path_finder.valid_gridpoints[int(self.path[-1])] != self.search_space.target).any():
            free_path = self.path_finder.valid_links[int(current_gridpoint_idx), 1:][self.path_finder.valid_links[int(current_gridpoint_idx), 1:] > 0]
            current_gridpoint_idx = random.choice(free_path)
            self.path = np.append(self.path, current_gridpoint_idx)

class GA:
    def __init__(self):
        self.search_space = Space()
        self.valid_gridpoints = self.search_space.valid_gridpoints
        self.population = []

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
        self.ga_set_dir = os.path.join(self.ga_folder, "Set_0")
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

    def distance(self, pt1, pt2):
        return np.sqrt(((pt2 - pt1) ** 2).sum())

    # lower fitness is better
    def ga_fitness(self, chromosome):
        path_length = 0
        for i in range(len(chromosome.path) - 1):
            pt1 = np.array(self.valid_gridpoints[int(chromosome.path[i])])
            pt2 = np.array(self.valid_gridpoints[int(chromosome.path[i+1])])
            path_length += self.distance(pt1, pt2)
        return path_length

    def get_genbest(self, elite_frac):
        self.gen_fitness = np.array([self.ga_fitness(chrom) for chrom in self.population])
        self.elites = []
        n_elites = math.ceil(elite_frac * len(self.population))
        elite_values = list(np.sort(list(set(heapq.nsmallest(n_elites, self.gen_fitness)))))
        # elite_chroms = np.array([np.where(gen_fitness == elite_value)[0] for elite_value in elite_values])
        for i in range(len(elite_values)):
            self.elites += list((np.where(self.gen_fitness == elite_values[i]))[0])
        self.best_fitness_history.append(self.gen_fitness[self.elites[0]])

    def generate_new_population(self, tournament_size, mutation_rate):
        new_population = []
        while len(new_population) != len(self.population):
            child1 = Chromosome(generate_path = False)
            child2 = Chromosome(generate_path = False)
            child1.path, child2.path = self.crossover(tournament_size)
            new_population += [child1, child2]
        # replace with new generation and mutate
        self.population = self.mutate(new_population, mutation_rate)

    def crossover(self, tournament_size):
        iter = 0
        while iter < 3:
            parent1 = self.tournament_selection(tournament_size).path
            parent2 = self.tournament_selection(tournament_size).path

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
        tournament = random.sample(range(len(self.population)), tournament_size)
        tournament_fitness = [self.gen_fitness[i] for i in tournament]
        best_chrom = int(tournament[tournament_fitness.index(min(tournament_fitness))])
        return self.population[best_chrom]

    def mutate(self, population, mutation_rate):
        for chrom in population:
            if random.random() < mutation_rate:
                mutation_idx = random.choice(range(len(chrom.path)))
                # delete all nodes after mutation point
                chrom.path = chrom.path[:mutation_idx + 1]
                # generate complete path from last node
                chrom.generate_path()
        return population

    def visualise_ga(self, generation):
        plt.figure(figsize=(5, 5))
        plt.scatter(self.search_space.obstacles[:,0], self.search_space.obstacles[:,1], marker='s', color="#000000")
        plt.scatter(self.search_space.target[0], self.search_space.target[1], marker='x', color="C1")
        color_idx = 0

        plt.annotate('Start Point', xy=(self.valid_gridpoints[0][0], self.valid_gridpoints[0][0]))
        plt.annotate('Goal Point', xy=(self.search_space.target[0], self.search_space.target[1]))

        plt.text(x=-0.5, y=-1.5, s='Generation:(%s)' % (generation))

        best_path_x = []
        best_path_y = []
        for point_idx in self.population[self.elites[0]].path:
            best_path_x.append(self.valid_gridpoints[int(point_idx)][0])
            best_path_y.append(self.valid_gridpoints[int(point_idx)][1])

        plt.plot(best_path_x, best_path_y, "g-")
        plt.xlim(-0.5, self.search_space.gridsize-0.5)
        plt.ylim(self.search_space.gridsize-0.5, -0.5)
        plt.draw()
        figname = "gen_" + str(generation) + ".png"
        plt.savefig(os.path.join(self.ga_set_dir, figname))
        plt.show()

def pso(n_robots=10, target_error=3.0, max_iterations=200, visualise=True):
    # Initialise PSO
    search_space = Space()
    robots_vector = [Robot() for _ in range(n_robots)]
    search_space.robots = robots_vector

    gbest_history = []
    iteration = 1
    while iteration < max_iterations:
        search_space.set_pbest()
        search_space.set_gbest()
        gbest_history.append(search_space.gbest_value)
        #print("Iter: ", iteration, "Best pos: ", search_space.gbest_position, "Best val: ", search_space.gbest_value)
        if visualise:
            search_space.visualise_pso()

        if search_space.gbest_value < target_error:
            break

        search_space.move_robots(w=0.7298, c1=3, c2=1.3)
        iteration += 1
    print("Target location: ", search_space.target)
    print("PSO best solution is: ", search_space.gbest_position, "in n_iterations: ", iteration)
    print("Target error: ", search_space.gbest_value)
    return iteration, gbest_history

def ga(population_size=100, elite_frac=0.05, tournament_size=5, mutation_rate=0.1, max_generations=20,
       max_consecutive_generations=8, visualise=True):
    # Initialise GA
    search_space = Space()
    path_finder = GA()
    chromosomes = [Chromosome(generate_path = True) for _ in range(population_size)]
    path_finder.population = chromosomes

    # GA loop
    generation = 1
    while generation < max_generations:
        path_finder.get_genbest(elite_frac)
        #print(path_finder.best_fitness_history[-1])

        if visualise:
            path_finder.visualise_ga(generation)

        if generation > max_consecutive_generations:
            if path_finder.best_fitness_history[-1] == path_finder.best_fitness_history[-max_consecutive_generations]:
                print("no new best solution found")
                break
        path_finder.generate_new_population(tournament_size, mutation_rate)
        generation += 1

    print("generations: ", generation)
    print("distance of shortest path found: ", path_finder.best_fitness_history[-1])
    return generation, path_finder.best_fitness_history

def main():
    # run once and visualise
    run_pso = False
    run_ga = False

    # change parameter values and plot graphs
    run_pso_param = False
    run_ga_param = True

    # PSO param search
    n_robots_range = range(5, 10, 1)
    pso_num_runs = 25

    # GA param search
    ga_popsize_range = range(10, 30, 10)
    ga_num_runs = 25

    if run_pso:
        iteration, gbest_history = pso(n_robots=10,
                                       target_error=3.0,
                                       max_iterations=200,
                                       visualise=False)

    if run_ga:
        generation, best_fitness_history = ga(population_size=100,
                                              elite_frac=0.05,
                                              tournament_size=5,
                                              mutation_rate=0.1,
                                              max_generations=20,
                                              max_consecutive_generations=8,
                                              visualise=True)

    if run_pso_param:
        n_robots_ls = []
        iter_ls = []
        iter_std_ls = []
        best_ls = []
        best_std_ls = []

        for n_robots in n_robots_range:
            param_iter_ls = []
            param_best_ls = []
            for i in range(pso_num_runs):
                print("n_robots: ", n_robots, "run: ", i)
                random.seed(i)
                iteration, gbest_history = pso(n_robots=n_robots,
                                               target_error=3.0,
                                               max_iterations=200,
                                               visualise=False)
                param_iter_ls.append(iteration)
                param_best_ls.append(gbest_history[-1])
            # save average and std to list
            n_robots_ls.append(n_robots)
            iter_ls.append(np.mean(param_iter_ls))
            iter_std_ls.append(np.std(param_iter_ls))
            best_ls.append(np.mean(param_best_ls))
            best_std_ls.append(np.std(param_best_ls))

        print(iter_std_ls, best_std_ls)
        # plot graphs
        plt.figure(0)
        ave = plt.plot(n_robots_ls, iter_ls, label="mean")
        std = plt.fill_between(n_robots_ls, np.array(iter_ls) - np.array(iter_std_ls),
                         np.array(iter_ls) + np.array(iter_std_ls), alpha = 0.3, label="std")
        plt.xlabel("No. of Robots")
        plt.ylabel("No. of Iterations")
        plt.title('No. of Iterations vs No. of robots')
        plt.legend()
        # plt.figure(1)
        # plt.plot(n_robots_ls, best_ls)
        # plt.fill_between(n_robots_ls, np.array(best_ls) - np.array(best_std_ls),
        #                  np.array(best_ls) + np.array(best_std_ls))
        # plt.xlabel("No. of Robots")
        # plt.ylabel("Error of Final ")
        # plt.title('No. of Iterations vs No. of robots')

        plt.draw()
        plt.show()


    if run_ga_param:
        pop_size_ls = []
        n_gen_ls = []
        n_gen_std_ls = []
        best_dist_ls = []
        best_dist_std_ls = []

        for pop_size in ga_popsize_range:
            param_gen_ls = []
            param_dist_ls = []
            for i in range(ga_num_runs):
                print("pop_size: ", pop_size, "run: ", i)
                random.seed(i)
                generation, best_fitness_history = ga(population_size=pop_size,
                                                      elite_frac=0.05,
                                                      tournament_size=5,
                                                      mutation_rate=0.1,
                                                      max_generations=20,
                                                      max_consecutive_generations=8,
                                                      visualise=False)
                param_gen_ls.append(generation)
                param_dist_ls.append(best_fitness_history[-1])
            # save average and std to list
            pop_size_ls.append(pop_size)
            n_gen_ls.append(np.mean(param_gen_ls))
            n_gen_std_ls.append(np.std(param_gen_ls))
            best_dist_ls.append(np.mean(param_dist_ls))
            best_dist_std_ls.append(np.std(param_dist_ls))

        print(n_gen_std_ls, best_dist_std_ls)

        # todo: consider plotting "no. of iterations = gen * pop size"?

        # plot graphs
        plt.figure(0)
        plt.plot(pop_size_ls, n_gen_std_ls, label="mean")
        plt.fill_between(pop_size_ls, np.array(n_gen_ls) - np.array(n_gen_std_ls),
                         np.array(n_gen_ls) + np.array(n_gen_std_ls), alpha=0.3, label="std")
        plt.xlabel("Population Size")
        plt.ylabel("No. of Generations")
        plt.title('No. of Generations vs Population size')

        plt.figure(1)
        plt.plot(pop_size_ls, best_dist_ls, label="mean")
        plt.fill_between(pop_size_ls, np.array(best_dist_ls) - np.array(best_dist_std_ls),
                         np.array(best_dist_ls) + np.array(best_dist_std_ls), alpha=0.3, label="std")
        plt.xlabel("Population Size")
        plt.ylabel("Length of Best Path")
        plt.title('No. of Generations vs Population size')

        plt.draw()
        plt.show()
if __name__ == '__main__':
    main()