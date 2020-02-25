# PSO adapted from: https://medium.com/analytics-vidhya/implementing-particle-swarm-optimization-pso-algorithm-in-python-9efc2eb179a6

import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from celluloid import Camera

dim = 2

class Robot:
    def __init__(self):
        self.search_space = Space()
        self.position = self.search_space.generate_item() #np.zeros(dim)
        self.velocity = np.random.randint(0, 10, dim)

        # particle best
        self.pbest_position = copy.deepcopy(self.position)
        self.pbest_value = float('inf')

        self.path = [[i] for i in self.position]

    def move(self):
        step = np.round(self.velocity)
        destination = self.position + step
        # print("position: ", self.position)
        # print("destination: ", destination)
        while True:
            # zero velocity
            if (step == 0).all():
                break
            else:
                print("step: ", step)
                next_position = self.position + np.round(step/max(abs(step)))
            print(self.position)
            print(next_position)

            # reached target
            if (next_position == self.search_space.target).all():
                self.position = next_position
                print("reached target")
                break
            # reached destination
            elif (next_position == destination).all():
                self.position = next_position
                print("reached point")
                break
            # hit wall
            elif (next_position >= self.search_space.gridsize * np.ones(dim)).any() or (next_position < np.zeros(dim)).any():
                self.velocity = -step #np.zeros(dim)
                print("hit wall")
                # print("hit wall")
                break
            # hit obstacle
            elif self.search_space.map[int(next_position[1]), int(next_position[0])] == 1:
                print("hit obstacle")
                # print(next_position)
                break
            # step into new position
            else:
                self.position = next_position
                step = destination - self.position
                print("continue")
        [self.path[i].append(self.position[i]) for i in range(len(self.path))]

# todo: obstacle map search space
class Space:
    def __init__(self):
        # load map
        self.map, self.map_plot = self._load_map()
        self.gridsize = self.map.shape[0]

        self.target = np.array([5, 17])#self.generate_item()
        self.robots = []
        # self.fig = plt.figure()
        # self.ax = plt.axes(xlim=(0, gridsize), ylim=(0, gridsize))
        # self.camera = Camera(self.fig)
        # self.line, = ax.plot([], [], lw=3)

        # global best
        self.gbest_position = np.random.uniform(0, self.gridsize, dim)
        self.gbest_value = float('inf')

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
            if self.map[item[1],item[0]] == 0:
                break
        return item

    # todo: generate fitness based on distance
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
        plt.ion()
        # x = [self.target[0]]
        # y = [self.target[1]]
        plt.scatter(self.map_plot[:,0],self.map_plot[:,1], marker='s')
        plt.scatter(self.target[0], self.target[1], marker='o')
        for robot in self.robots:
            # x.append(robot.position[0])
            # y.append(robot.position[1])
            plt.plot(robot.path[0], robot.path[1], marker="x")
        # plt.scatter(x, y)
        plt.xlim(0, self.gridsize-1)
        plt.ylim(self.gridsize-1, 0)
        plt.grid(True)
        plt.show()

def main():
    # Input
    # n_iterations = int(input("Inform the number of iterations: "))
    # target_error = float(input("Inform the target error: "))
    # n_robots = int(input("Inform the number of robots: "))

    n_iterations = 100
    target_error = 1.0
    n_robots = 15

    search_space = Space()
    robots_vector = [Robot() for _ in range(n_robots)]
    search_space.robots = robots_vector
    print("target location: ", search_space.target)

    # fig = plt.figure()
    # camera = Camera(fig)

    iteration = 0
    while iteration < n_iterations:
        search_space.set_pbest()
        search_space.set_gbest()

        print("Iter: ", iteration, "Best pos: ", search_space.gbest_position, "Best val: ", search_space.gbest_value)

        search_space.visualise()
        # search_space.camera.snap()
        # anim = search_space.camera.animate()
        # x = [target[0]]
        # y = [target[1]]
        # fig = plt.plot(search_space.target[0], search_space.target[1], marker='o')
        # for robot in search_space.robots:
        #     # x.append(robot.position[0])
        #     # y.append(robot.position[1])
        #     fig = plt.plot(robot.position[0], robot.position[1], marker="x")

        # plt.xlim(0, gridsize)
        # plt.ylim(gridsize, 0)
        # plt.grid(True)
        # plt.show()
        # camera.snap()

        if search_space.gbest_value < target_error:
            break

        search_space.move_robots(w=0.7298, c1=3, c2=2)
        iteration += 1

    print("best solution is: ", search_space.gbest_position, "in n_iterations: ", iteration)
    # anim = camera.animate()
    # anim.save('test.mp4')

if __name__ == '__main__':
    main()