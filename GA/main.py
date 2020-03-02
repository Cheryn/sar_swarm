#!/usr/bin/env python

"""
Main script

Contains: main function

Author: Yasim Ahmad(yaaximus)

Email: yasim.ahmed63@yahoo.com
"""
from tools.population import create_population, _both_equ
from tools.fitness import fitness
from tools.ranking import ranking
from tools.dna import breed_by_crossover, _do_mutation
from tools.draw_plot import show_plot
from tools.selection import tournament_selection

from config import Config 

import numpy as np

def main():
    """
    This function encapsulates the cappbilty of initializing chromosome population
    and then continue calculating fitness, generate ranking, perform crossover and
    mutation & repeating above steps until a defined stopping crietia is not met.
    """
    #need to load the map to get path points
    Config.load_map()
    
    #create initial population
    chr_population = create_population()
    
    #specify available neighbours that can be traveled through
    link = Config.define_links()

    #calculate population fitness and specify the index of the best fit individual so
    #that it can be accessed for printing'
    chr_pop_fitness, chr_best_fitness_index = fitness(chr_pop=chr_population)
    
    #create a list to store all values of highest fit for each iteration
    best_fitness_history = []
    
    
    #this continuously loops
    while not Config.stop_generation:
        prev_best_fit = chr_pop_fitness[chr_best_fitness_index[0], 0]
        
        # Create an empty list for new population
        new_population=[]
        
    # Create new popualtion generating two children at a time (this combines selection and crossover
    #in one function)
        for i in range(int(len(chr_population)/2)):
            child_1, child_2 = breed_by_crossover(chr_population, chr_pop_fitness)
            new_population.append(child_1)
            new_population.append(child_2)
            
            #print(len(new_population))
    
    # Replace the old population with the new one
        chr_population = np.array(new_population)
        
    # Apply mutation
        
        chr_population = _do_mutation(chr_population, link)
    
        #caculate fitness for new population
        chr_pop_fitness, chr_best_fitness_index = fitness(chr_pop=chr_population)
        
        
        #termination condition is if the highest fit does not change for a given number of iterations
        if prev_best_fit == chr_pop_fitness[chr_best_fitness_index[0], 0]:
            Config.stop_criteria += 1
            
        else:
            Config.stop_criteria = 0

        if Config.stop_criteria >= 5:
            Config.stop_generation = True
        
        #print best path and its fitness
        print("Best chromosome is:", new_population[chr_best_fitness_index[0]])
        
        best_fitness = np.amax(chr_pop_fitness)
        print("Fitness_score is:", best_fitness)
        
        best_fitness_history.append(best_fitness)
        #plot best path
        show_plot(best_chromosome=new_population[0])
        
        #update generation number
        Config.generations += 1
        
        #new population becomes previous population in next loop
        chr_population = new_population
        
    #at the end print best fitness history
    print(best_fitness_history)
    
    
if __name__ == '__main__':

    
    
    main()
