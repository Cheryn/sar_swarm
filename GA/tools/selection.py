# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:47:04 2020

@author: chife
"""


from config import Config

import numpy as np
import random


def tournament_selection(population, scores):
    # Get population size
    population_size = int(Config.pop_max)
    tournament_size = Config.tournament_size
    
    # Pick individuals for tournament
    for i in range(population_size):
        tournament = random.sample(range(0, population_size-1), tournament_size)
    
        tournament_fitness = []
    # Get fitness score for each
        for j in tournament:
            fitness_score = scores[j]
            tournament_fitness.append(fitness_score)
    
        max_index = tournament_fitness.index(max(tournament_fitness))
        winner= int(tournament[max_index])
        
    # Return the chromsome of the winner
    return population[winner]


    
    