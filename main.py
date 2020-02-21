# PSO adapted from: https://medium.com/analytics-vidhya/implementing-particle-swarm-optimization-pso-algorithm-in-python-9efc2eb179a6
#
import numpy as np
import matplotlib.pyplot as plt
import copy

dim = 2
gridsize = 100


class Robot:
    def __init__(self):
        self.position = np.zeros(dim)
        self.velocity = np.random.uniform(0, 10, dim)

        # particle best
        self.pbest_position = copy.deepcopy(self.position)
        self.pbest_value = float('inf')

    # todo: wall restriction
    # todo: obstacle avoidance
    def move(self):
        self.position += self.velocity
        # if (self.position + self.velocity > 100 * np.ones(self.position.size)).any() or (self.position > -100 * np.ones(self.position.size)).any():
        #     self.position -= 2*self.velocity

# todo: obstacle map search space
class Space:
    def __init__(self, target, target_error, n_robots):
        self.target = target
        self.target_error = target_error
        self.n_robots = n_robots
        self.robots = []

        # global best
        self.gbest_position = np.random.uniform(0, gridsize, dim)
        self.gbest_value = float('inf')

    # todo: generate fitness based on distance
    def fitness(self, robot):
        return np.sum((self.target - robot.position)**2)**0.5
        # return robot.position[0] ** 2 + robot.position[1] ** 2 + 1

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
            robot.velocity = w * robot.velocity + \
                           c1 * np.random.uniform(0, 1) * (robot.pbest_position - robot.position) + \
                           c2 * np.random.uniform(0, 1) * (self.gbest_position - robot.position)
            robot.move()

    def visualise(self):
        plt.ion()
        plt.figure()
        plt.scatter(self.target[0], self.target[1], marker="o")
        for robot in self.robots:
            plt.scatter(robot.position[0], robot.position[1], marker="x")
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

    target = np.random.uniform(0, gridsize, dim)
    print("target location: ", target)

    search_space = Space(target, target_error, n_robots)
    robots_vector = [Robot() for _ in range(search_space.n_robots)]
    search_space.robots = robots_vector

    iteration = 0
    while iteration < n_iterations:
        search_space.set_pbest()
        search_space.set_gbest()

        print("Iter: ", iteration, "Best pos: ", search_space.gbest_position, "Best val: ", search_space.gbest_value)
        search_space.visualise()

        if search_space.gbest_value < search_space.target_error:
            print(search_space.gbest_value)
            print(search_space.gbest_position)
            break

        search_space.move_robots(w=0.7, c1=2, c2=2)
        iteration += 1

    print("best solution is: ", search_space.gbest_position, "in n_iterations: ", iteration)


if __name__ == '__main__':
    main()