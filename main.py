# PSO adapted from: https://medium.com/analytics-vidhya/implementing-particle-swarm-optimization-pso-algorithm-in-python-9efc2eb179a6

import os
import numpy as np
import copy
import matplotlib.pyplot as plt

dim = 2

class Robot:
    def __init__(self):
        self.search_space = Space()
        self.position = np.zeros(dim) #self.search_space.generate_item() #
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
        self.map, self.map_plot = self._load_map()
        self.gridsize = self.map.shape[0]

        self.target = (self.gridsize - 1) * np.ones(dim) #self.generate_item()
        self.robots = []

        # global best
        self.gbest_position = np.random.uniform(0, self.gridsize, dim)
        self.gbest_value = float('inf')

        # saving figures
        self.pso_folder = "PSO_plots"
        if not os.path.exists(self.pso_folder):
            os.makedirs(self.pso_folder)
        self.pso_set_dir = os.path.join(self.pso_folder, "set_0")
        if not os.path.exists(self.pso_set_dir):
            os.makedirs(self.pso_set_dir)
        self.fignum = 0

    def _load_map(self):
        map_filename = None
        txt_files = list(np.sort([file for file in os.listdir(os.getcwd()) if file.endswith(".txt")]))
        if len(txt_files) > 0:
            for txt_file in txt_files:
                map_filename = os.path.join(os.getcwd(), txt_file)

            if map_filename is None:
                print("no map found")
                return None
            else:
                with open(map_filename, "r") as file:
                    map_data = file.readlines()
                map_matrix = np.loadtxt(map_data)
                map_plot = []
                for y in range(len(map_matrix)):
                    for x in range(len(map_matrix[y])):
                        if map_matrix[y, x] == 1:
                            map_plot.append([x, y])
                return map_matrix, np.array(map_plot)

    def generate_item(self):
        while True:
            item = np.random.randint(self.map.shape[0], size=dim)
            if self.map[item[1], item[0]] == 0:
                break
        return item

    def fitness(self, robot):
        return np.sum((self.target - robot.position)**2)**0.5

    def set_pbest(self):
        for robot in self.robots:
            fitness_candidate = self.fitness(robot)
            if fitness_candidate < robot.pbest_value:
                robot.pbest_value = copy.deepcopy(fitness_candidate)
                robot.pbest_position = copy.deepcopy(robot.position)

    def set_gbest(self):
        for robot in self.robots:
            best_fitness_candidate = self.fitness(robot)
            if best_fitness_candidate < self.gbest_value:
                self.gbest_value = copy.deepcopy(best_fitness_candidate)
                self.gbest_position = copy.deepcopy(robot.position)

    # todo: gridwise movements
    def move_robots(self, w, c1, c2):
        for robot in self.robots:
            robot.velocity = w * (robot.velocity +
                           c1 * np.random.uniform(0, 1) * (robot.pbest_position - robot.position) +
                           c2 * np.random.uniform(0, 1) * (self.gbest_position - robot.position))
            robot.move()

    def visualise(self):
        #plt.ion()
        plt.figure(figsize=(5, 5))
        plt.scatter(self.map_plot[:,0], self.map_plot[:,1], marker='s', color="#000000")
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

def main():
    # Input
    # n_iterations = int(input("Inform the number of iterations: "))
    # target_error = float(input("Inform the target error: "))
    # n_robots = int(input("Inform the number of robots: "))

    n_iterations = 100
    target_error = 3.0
    n_robots = 5

    search_space = Space()
    robots_vector = [Robot() for _ in range(n_robots)]
    search_space.robots = robots_vector
    print("target location: ", search_space.target)

    iteration = 0
    while iteration < n_iterations:
        search_space.set_pbest()
        search_space.set_gbest()

        print("Iter: ", iteration, "Best pos: ", search_space.gbest_position, "Best val: ", search_space.gbest_value)

        search_space.visualise()

        if search_space.gbest_value < target_error:
            break

        search_space.move_robots(w=0.7298, c1=2.05, c2=2.05)
        iteration += 1

    print("best solution is: ", search_space.gbest_position, "in n_iterations: ", iteration)

if __name__ == '__main__':
    main()