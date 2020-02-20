import numpy as np
import matplotlib.pyplot as plt
import copy

w = 0.7
c1 = 2
c2 = 2


class Robot:
    def __init__(self, dim, minx, maxx):
        self.position = np.random.uniform(low=minx, high=maxx, size=dim)
        self.velocity = np.random.uniform(low=-10, high=10, size=dim)

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
    def __init__(self, dims, target, target_error, n_robots):
        self.target = target
        self.target_error = target_error
        self.n_robots = n_robots
        self.robots = []

        # global best
        self.gbest_position = np.random.uniform(low=-100, high=100, size=dims)
        self.gbest_value = float('inf')

    # todo: generate fitness based on distance
    def fitness(self, robot, target):
        return np.sum((target - robot.position)**2)**0.5
        # return robot.position[0] ** 2 + robot.position[1] ** 2 + 1

    def set_pbest(self):
        for robot in self.robots:
            fitness_candidate = self.fitness(robot, self.target)
            if fitness_candidate < robot.pbest_value:
                robot.pbest_value = copy.deepcopy(fitness_candidate)
                robot.pbest_position = copy.deepcopy(robot.position)

    def set_gbest(self):
        for robot in self.robots:
            best_fitness_candidate = self.fitness(robot, self.target)
            if best_fitness_candidate < self.gbest_value:
                self.gbest_value = copy.deepcopy(best_fitness_candidate)
                self.gbest_position = copy.deepcopy(robot.position)

    # todo: gridwise movements
    def move_robots(self):
        for robot in self.robots:
            robot.velocity = w * robot.velocity + \
                           c1 * np.random.uniform(0,1) * (robot.pbest_position - robot.position) + \
                           c2 * np.random.uniform(0,1) * (self.gbest_position - robot.position)

            robot.move()

    def visualise(self):
        plt.figure()
        plt.scatter(self.target[0], self.target[1], marker="o")
        for robot in self.robots:
            plt.scatter(robot.position[0], robot.position[1],marker="x")
        plt.xlim(-100,100)
        plt.ylim(-100,100)
        plt.show()


def main():
    # Input
    # n_iterations = int(input("Inform the number of iterations: "))
    # target_error = float(input("Inform the target error: "))
    # n_robots = int(input("Inform the number of robots: "))

    n_iterations = 100
    target_error = 0.1
    n_robots = 10

    target = np.random.uniform(low=-100, high=100, size=2)
    print("target location: ", target)

    search_space = Space(2, target, target_error, n_robots)
    robots_vector = [Robot(2, -100, 100) for _ in range(search_space.n_robots)]
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

        search_space.move_robots()
        iteration += 1

    print("best solution is: ", search_space.gbest_position, "in n_iterations: ", iteration)


if __name__ == '__main__':
    main()