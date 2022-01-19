# Script to perform multi-objective optimization of an insect chain, using only one type of insect
#

import inspyred # library for evolutionary algorithms
import io
import numpy as np
import os
import random # to generate random numbers
import sys

from json import load, dump

def load_instance(json_file):
    """
    Inputs: path to json file
    Outputs: json file object if it exists, or else returns NoneType
    """
    if os.path.exists(path=json_file):
        with io.open(json_file, 'rt', newline='') as file_object:
            return load(file_object)
    return None

def fitness_function(candidate, json_instance) :

    return


def generator(random, args) :

    boundaries = args["boundaries"]

    # here we need to generate a random individual and check that the boundaries are respected

    return

def evaluator(candidates, args) :

    list_of_fitness_values = []
    for candidate in candidates :
        f1, f2 = fitness_function(candidate)
        list_of_fitness_values.append(inspyred.ec.emo.Pareto( [f1, f2] )) # in this case, for multi-objective optimization we need to create a Pareto fitness object with a list of values

    return list_of_fitness_values

def variator(random, candidates, args) :

    # after mutation or cross-over, check that the individual is still valid

    return

def observer(population, num_generations, num_evaluations, args) :

    print("Generation %d (%d evaluations)" % (num_generations, num_evaluations))

    return

def main() :

    # a few hard-coded parameters
    random_seed = 42

    # load information on the problem
    json_instance = load_instance('../data/insect_data.json') 

    # boundaries for all the values included in the individual
    boundaries = dict() # dict or list?

    # initialize random number generator
    random_number_generator = random_number_generator = random.Random()
    random_number_generator.seed(random_seed)

    # create instance of NSGA2
    nsga2 = inspyred.ec.emo.NSGA2(random_number_generator)
    nsga2.terminator = inspyred.ec.terminators.evaluation_termination # stop after a certain number of evaluations
    nsga2.variator = [inspyred.ec.variators.blend_crossover, inspyred.ec.variators.gaussian_mutation] # types of evolutionary operators to be used

    final_pareto_front = nsga2.evolve(
                            generator = generator,
                            evaluator = evaluator,
                            pop_size = 100,
                            num_selected = 200,
                            max_evaluations = 2000,
                            maximize = True, # it's a minimization problem

                            # all arguments specified below, THAT ARE NOT part of the "evolve" method, will be automatically placed in "args"
                            # "args" is a dictionary that is passed to all functions
                            boundaries = boundaries,
                            json_instance = json_instance,
    )


    return

# calls the 'main' function when the script is called
if __name__ == "__main__" :
    sys.exit( main() )
