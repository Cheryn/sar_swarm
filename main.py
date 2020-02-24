# PSO adapted from: https://medium.com/analytics-vidhya/implementing-particle-swarm-optimization-pso-algorithm-in-python-9efc2eb179a6
#
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from celluloid import Camera

dim = 2
gridsize = 20


class Robot:
    def __init__(self):
        self.position = np.zeros(dim)
        self.velocity = np.random.uniform(0, 10, dim)

        # particle best
        self.pbest_position = copy.deepcopy(self.position)
        self.pbest_value = float('inf')

        self.path = [[i] for i in self.position]

    # todo: obstacle avoidance
    def move(self):
        next_position = self.position + self.velocity
        # wall restriction
        if (next_position < gridsize * np.ones(dim)).all() and (next_position > np.zeros(dim)).all():
            self.position = next_position
        [self.path[i].append(self.position[i]) for i in range(len(self.path))]

# todo: obstacle map search space
class Space:
    def __init__(self, target_error, n_robots):
        # load map
        self.map = self._load_map()

        self.target = self.generate_target()
        self.target_error = target_error
        self.n_robots = n_robots
        self.robots = []
        self.fig = plt.figure()
        # self.ax = plt.axes(xlim=(0, gridsize), ylim=(0, gridsize))
        self.camera = Camera(self.fig)
        # self.line, = ax.plot([], [], lw=3)

        # global best
        self.gbest_position = np.random.uniform(0, gridsize, dim)
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
                return np.loadtxt(map_data)

    def generate_target(self):
        while True:
            target = np.random.randint(self.map.shape[0], size=dim)
            if self.map[target[0],target[1]] == 0:
                break
        return target

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
            robot.velocity = w * (robot.velocity + \
                           c1 * np.random.uniform(0, 1) * (robot.pbest_position - robot.position) + \
                           c2 * np.random.uniform(0, 1) * (self.gbest_position - robot.position))
            robot.move()

    def visualise(self):
        # plt.ion()
        x = [self.target[0]]
        y = [self.target[1]]
        # self.fig = plt.scatter(self.target[0], self.target[1], marker='o')
        for robot in self.robots:
            x.append(robot.position[0])
            y.append(robot.position[1])
            # self.fig = plt.scatter(robot.position[0], robot.position[1], marker="x")
        plt.scatter(x, y)
        plt.xlim(0, gridsize)
        plt.ylim(0, gridsize)
        plt.show()

def main():
    # Input
    # n_iterations = int(input("Inform the number of iterations: "))
    # target_error = float(input("Inform the target error: "))
    # n_robots = int(input("Inform the number of robots: "))

    n_iterations = 100
    target_error = 0.1
    n_robots = 20

    search_space = Space(target_error, n_robots)
    robots_vector = [Robot() for _ in range(search_space.n_robots)]
    search_space.robots = robots_vector
    print("target location: ", search_space.target)

    fig = plt.figure()
    camera = Camera(fig)

    iteration = 0
    while iteration < n_iterations:
        search_space.set_pbest()
        search_space.set_gbest()

        print("Iter: ", iteration, "Best pos: ", search_space.gbest_position, "Best val: ", search_space.gbest_value)

        # search_space.visualise()
        # search_space.camera.snap()
        # anim = search_space.camera.animate()
        # x = [target[0]]
        # y = [target[1]]
        fig = plt.scatter(search_space.target[0], search_space.target[1], marker='o')
        for robot in search_space.robots:
            # x.append(robot.position[0])
            # y.append(robot.position[1])
            fig = plt.scatter(robot.position[0], robot.position[1], marker="x")

        plt.xlim(0, gridsize)
        plt.ylim(0, gridsize)
        plt.show()
        camera.snap()

        if search_space.gbest_value < search_space.target_error:
            break

        search_space.move_robots(w=0.7298, c1=3, c2=2)
        iteration += 1

    print("best solution is: ", search_space.gbest_position, "in n_iterations: ", iteration)
    anim = camera.animate()
    anim.save('test.mp4')

if __name__ == '__main__':
    main()