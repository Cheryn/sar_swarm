#!/usr/bin/env python

"""
Script containing functionality related to DNA in Genetics like crossover & mutation.

Contains: 'do crossover' & 'do mutation' function

Author: Yasim Ahmad(yaaximus)

Email: yasim.ahmed63@yahoo.com
"""

from config import Config

import numpy as np
import random

from tools.population import _both_equ
from tools.selection import tournament_selection

#this function is not used but not sure if we need it
def dna(chr_pop_fitness, ranked_population, chr_best_fitness_index, last_pop):
    """
    This function encapsulates functionality related to dna like crossover
    and mutation.

    Parameters
    ----------
    chr_pop_fitness : [numpy.ndarray]
        [Contains fitness values of chromosome population]
    ranked_population : [numpy.ndarray]
        [Contains numpy array of ranked chromosome population]
    chr_best_fitness_index : [list]
        [Contains list of best fitness indices in chromosome population]
    last_pop : [numpy.ndarray]
        [Contains numpy array of last population]

    Returns
    -------
    [numpy.ndarray]
        [numpy array of chromosome with have gone through random crossover and mutation]
    """

    chromo_crossover_pop = _do_crossover(
        ranked_pop=ranked_population, chr_best_fit_indx=chr_best_fitness_index,
        pop=last_pop)

    chromo_crossover_mutated_pop = _do_mutation(pop=chromo_crossover_pop)

    return chromo_crossover_mutated_pop


def _do_mutation(population,link):
    """
    This function is responsible for handling mutation in population of chromosomes.
    This loops through the pahts and randomly calculates a number, a and 
    if it is less than the mutation rate, the path is mutated by choosing a random node and
    changing it and generating a whole new path onwards from that node
    
    
    Parameters
    ----------
    pop : [numpy.ndarray]
        [numpy array of chromosome population which will undergo mutation]

    Returns
    -------
    [numpy.ndarray]
        [numpy array of chromosome population undergone mutation]
    """
    
    mutated_pop = np.array(population, copy=True)
    
    #itr = 3
    #while itr < Config.pop_max:
    for i in range(len(population)):
        a = random.random()
        #for each path generate a random number and compare with the mutation_probablity
        
        chromosome = population[i]
            
        if a <= Config.mutation_rate:
            #specify mutation point and its index in the path
            mutation_point = int(random.choice(chromosome))
            
            mutation_index = np.where(chromosome==mutation_point)[0][0]
            
            #find a new point to replace the mutation point
            free_path = link[mutation_point,1:-1][link[mutation_point,1:-1]>0]
            
            
            new_node = random.choice(free_path)
            
            #replace mutation point
            chromosome[mutation_index] = new_node
            
            
            #generate new path
            x= mutation_index #dummy variable
            j= mutation_index+1
                
            r= new_node+1
            
            while j < Config.chr_len:
	       
                n=int(r-1)
                
                
                free_path = link[n][link[n]>0]
        
                next_point = random.choice(free_path)
                if next_point != n:
                    chromosome[j] = next_point
                else: 
                    j-=1
                x = next_point
          
           
                if _both_equ(x,Config.end_index):
                    while j < (Config.chr_len-1):
                        chromosome[j+1] = Config.end_index
                        j+=1
                        
                r = next_point+1
                j+=1
        
        #itr += 1
    return mutated_pop
    
""" 


    itr = 3
    while itr < Config.pop_max:
        for k in range(Config.chr_len):
            c = random.random()
            if c < Config.mutation_rate and k is not 0:
                mutated_pop[itr, k] = random.randint(1, Config.npts-2)
            else:
                pass
        itr += 1
    return mutated_pop
"""

def _do_crossover(parent1, parent2):
    """
    This function is responsible for handling crossover in population of chromosomes.

    Parameters
    ----------
    ranked_pop : [numpy.ndarray]
        [numpy array of chromosome population which will undergo crossover]
    chr_best_fit_indx : [list]
        [Contains list of best fitness indices in chromosome population]
    pop : [numpy.ndarray]
        [numpy array of chromosome population to get best fitness chromosomes
         from last population]

    Returns
    -------
    [numpy.ndarray]
        [numpy array of chromosome population undergone crossover]
    """
    
def breed_by_crossover(population,scores):
    
    
    itr = 0 
    
    while itr<3:
        parent_1 = tournament_selection(population, scores)
        parent_2 = tournament_selection(population, scores)
    # Pick crossover point, avoding ends of chromsome
        common_points = np.intersect1d(parent_1, parent_2)
    
    #chack that there is a 
        if len(common_points)>0:
            crossover_point = random.choice(common_points)
            idx1 = np.where(parent_1==crossover_point)[0][0]
            idx2 = np.where(parent_2==crossover_point)[0][0]
            break
    
    #if both are the same just crossover at the start point ie they remain the same
        else:
            idx1 = 0
            idx2 = 0
        itr+=1
    # Create children. np.hstack joins two arrays
    if idx1 == 0:
        child_1 = parent_1
        child_2 = parent_2
    else:
        child_1 = np.hstack((parent_1[0:idx1],
                        parent_2[idx2:]))
        #print(child_1.shape,'199')
        child_2 = np.hstack((parent_2[0:idx2],
                        parent_1[idx1:]))
        #print(child_2.shape,'198')
        if len(child_1) > Config.chr_len:
            child_1 = child_1[:Config.chr_len]
            #print(child_1.shape,'197')
        elif len(child_1)<=Config.chr_len:
            added_end1 = np.full((Config.chr_len - len(child_1)),Config.end_index)
            #print(added_end.shape)
            child_1 = np.hstack((child_1,added_end1))
            #print(child_1.shape,'202')
        if len(child_2) > Config.chr_len:
            child_2 = child_2[:Config.chr_len]
            #print(child_2.shape,'205')
        elif len(child_2)<=Config.chr_len:
            added_end2 = np.full((Config.chr_len - len(child_2)),Config.end_index)
            #print(added_end.shape)
            child_2 = np.hstack((child_2,added_end2))
            #print(child_2.shape,'209')
            
    # Return children
    return child_1, child_2

"""
    crossover_pop = np.zeros((Config.pop_max, Config.chr_len))

    crossover_pop[0, :] = pop[chr_best_fit_indx[0], :]
    crossover_pop[1, :] = pop[chr_best_fit_indx[1], :]
    crossover_pop[2, :] = pop[chr_best_fit_indx[2], :]

    itr = 5

    while itr < Config.pop_max / 5:

        a = random.randint(0, Config.chr_len - 1)
        b = random.randint(0, Config.chr_len - 1)
        print(a)
        partner_a = ranked_pop[a, :]
        partner_b = ranked_pop[b, :]
        print(b)
        joining_pt = random.randint(0, Config.chr_len - 1)

        crossover_pop[itr, :joining_pt] = partner_a[:joining_pt]
        crossover_pop[itr+1, :joining_pt] = partner_b[:joining_pt]

        crossover_pop[itr, joining_pt:] = partner_b[joining_pt:]
        crossover_pop[itr+1, joining_pt:] = partner_a[joining_pt:]

        itr += 2

    while itr < Config.pop_max:

        crossover_pop[itr] = ranked_pop[itr]
        itr += 1

    return crossover_pop
"""