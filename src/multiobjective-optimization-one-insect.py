# Script to perform multi-objective optimization of an insect chain, using only one type of insect
#

import copy
import inspyred # library for evolutionary algorithms
import io
import numpy as np
import os
import pandas as pd # to manipulate CSV files
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

    return 0, 0, 0 # TODO remove this, it's just to perform some trial runs

    operating_profit = 0 # maximize
    insect_frass = 0 # minimize
    labor = 0 # maximize
    labor_safety = 0 # maximize
    for feed_id in range(len(json_instance['feed'])):
            operating_profit += (json_instance["C"]-json_instance["labor"]*candidate.Number_of_employee-json_instance["feed_cost"]-json_instance["energy"]-json_instance["rent"])*json_instance["FCE"][feed_id]*candidate.Amount_feed[feed_id]
            for scale_id in len(json_instance['scales']):
                insect_frass += candidate.Amount_feed[feed_id]/json_instance["NRF"][feed_id]*(1-json_instance["FCE"][feed_id])*json_instance["NRF"][feed_id]*json_instance["SF_ls"][scale_id]*candidate.scale[scale_id]
                 
    labor = json_instance["RW"]*json_instance["RWT"]*json_instance["CFfw"]*candidate.Number_of_employee
    for equipment_id in range(len(json_instance['equipments'])):
        for scale_id in len(json_instance['scales']):
            labor_safety += json_instance["PPE"][equipment_id]*candidate.equipments[equipment_id]*json_instance["SFls"][scale_id]*candidate.scale[scale_id]*candidate.Number_of_employee
    

    return operating_profit, insect_frass, labor, labor_safety


def generator(random, args) :

    boundaries = args["boundaries"]
    
    # here we need to generate a random individual and check that the boundaries are respected
    # to (hopefully) make our lives easier, individuals are encoded as dictionaries
    print("Generating new individual...")
    individual = dict()

    # first step: randomize the scale of the company
    individual["SC"] = random.choice(boundaries["SC"])

    # other values are in (0,1), and will then be scaled depending on the scale of the company (before evaluation)
    individual["AIF"] = random.uniform(0, 1)
    individual["Nl"] = random.uniform(0, 1)

    # protective equipments can or cannot be acquired
    individual["EQ"] = list()
    for i in range(0, boundaries["EQ"]) :
        individual["EQ"].append(random.choice([0, 1]))

    # types of feed: they are encoded as a list of floats, that has to be normalized (they represent percentages)
    individual["F"] = list()
    for i in range(0, boundaries["F"]) :
        individual["F"].append(random.uniform(0, 1))
    denominator = sum(individual["F"])

    for i in range(0, boundaries["F"]) :
        individual["F"][i] /= denominator

    # some output
    print("Individual generated: %s" % str(individual))

    return individual

def evaluator(candidates, args) :

    json_instance = args["json_instance"]

    list_of_fitness_values = []
    for candidate in candidates :
        f1, f2, f3 = fitness_function(candidate, json_instance)
        list_of_fitness_values.append(inspyred.ec.emo.Pareto( [f1, f2, f3] )) # in this case, for multi-objective optimization we need to create a Pareto fitness object with a list of values

    return list_of_fitness_values

@inspyred.ec.variators.crossover
def variator(random, candidate1, candidate2, args) :

    children = []
    boundaries = args["boundaries"]

    # decide whether we are going to perform a cross-over
    perform_crossover = random.uniform(0, 1) < 0.8 # this is True or False

    # cross-over
    if perform_crossover :
        print("I am going to perform a cross-over!")
        child1 = copy.deepcopy(candidate1)
        child2 = copy.deepcopy(candidate2)

        # for every key in the dictionary (every part of the individual)
        # randomly swap everything with a 50% probability
        # TODO maybe do something better with values associated to lists (EQ, F) ?
        for k in child1 :
            if random.uniform(0, 1) < 0.5 :
                temp = child1[k]
                child1[k] = child2[k]
                child2[k] = temp

        # append children
        children = [child1, child2]

    # mutation(s)
    # if there are no children, create a new one
    if len(children) == 0 :
        children.append(copy.deepcopy(candidate1))

    # randomly choose which part of the individual we are going to mutate
    for individual in children :
        to_be_mutated = random.choice([k for k in individual])
        print("I am going to mutate part \"%s\"" % to_be_mutated)

        # different cases
        if to_be_mutated == "SC" :
            # pick another scale for the company, different from the current one
            sc_choices = [sc for sc in boundaries["SC"] if sc != individual["SC"]]
            individual["SC"] = random.choice(sc_choices)

        elif to_be_mutated == "AIF" or to_be_mutated == "Nl" :
            # modify the quantity in (0,1) with a small Gaussian mutation
            individual[to_be_mutated] += random.gauss(0, 0.1)

        elif to_be_mutated == "EQ" :
            # this is easy, perform a random number of bit flips; low (high probability) or high (low probability)
            number_of_bit_flips = min(random.randint(1, len(individual["EQ"])+1) for i in range(0, len(individual["EQ"])))
            # choose several equipments (with replacement)
            indexes = random.sample(range(0, len(individual["EQ"])), number_of_bit_flips)

            for index in indexes :
                if individual["EQ"][index] == 0 :
                    individual["EQ"][index] = 1
                else :
                    individual["EQ"][index] = 0

        elif to_be_mutated == "F" :
            # perform a random number of value modifications; low (high probability) or high (low probability)
            number_of_modifications = min(random.randint(1, len(individual["F"])+1) for i in range(0, len(individual["F"])))
            # choose several types of feed (with replacement)
            indexes = random.sample(range(0, len(individual["F"])), number_of_modifications)

            # small Gaussian mutation on each quantity
            for index in indexes :
                individual["F"][index] += random.gauss(0, 0.1)

    # after mutation or cross-over, check that the individual is still valid
    # in our case, we just need to normalize the amounts of each type of feed, and check that the
    # quantities in (0,1) are still in (0,1)
    for individual in children :

        denominator = sum(individual["F"])
        for i in range(0, boundaries["F"]) :
            individual["F"][i] /= denominator

        for q in ["AIF", "Nl"] :
            if individual[q] > 1.0 :
                individual[q] = 1.0
            elif individual[q] < 0.0 :
                individual[q] = 0.0

    return children

def observer(population, num_generations, num_evaluations, args) :

    print("Generation %d (%d evaluations)" % (num_generations, num_evaluations))

    return

def main() :

    # a few hard-coded parameters
    random_seed = 42

    # TODO also, we should do things properly and create a log file

    # load information on the problem
    json_instance = load_instance('../data/insect_data.json') 

    # boundaries for all the values included in the individual
    boundaries = dict()
    boundaries["SC"] = [1, 2, 3, 4] # minimum and maximum
    boundaries["EQ"] = 5 # number of different types of equipments
    boundaries["F"] = 5 # types of different feeds

    # TODO boundaries for AIF and Nl, depending on SC

    # initialize random number generator
    random_number_generator = random_number_generator = random.Random()
    random_number_generator.seed(random_seed)

    # create instance of NSGA2
    nsga2 = inspyred.ec.emo.NSGA2(random_number_generator)
    nsga2.observer = observer
    nsga2.terminator = inspyred.ec.terminators.evaluation_termination # stop after a certain number of evaluations
    nsga2.variator = [variator] # types of evolutionary operators to be used

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

    # save the final Pareto front in a .csv file
    # prepare dictionary that will be later converted to .csv file using Pandas library
    df_dictionary = { "SC": [], "AIF": [], "Nl": [] } 
    for e in range(0, boundaries["EQ"]) :
        df_dictionary["EQ" + str(e)] = []
    for f in range(0, boundaries["F"]) :
        df_dictionary["F" + str(f)] = []

    # TODO change names of the fitnesses to their appropriate correspondence (e.g. "Profit", "Social Impact", "Environmental Impact")
    df_dictionary["Fitness1"] = []
    df_dictionary["Fitness2"] = []
    df_dictionary["Fitness3"] = []

    # go over the list of individuals in the Pareto front and store them in the dictionary 
    for individual in final_pareto_front :

        genome = individual.candidate
        for k in genome :
            # manage parts of the genome who are lists
            if isinstance(genome[k], list) :
                for i in range(0, len(genome[k])) :
                    df_dictionary[k + str(i)].append(genome[k][i])
            else :
                df_dictionary[k].append(genome[k])

        df_dictionary["Fitness1"] = individual.fitness[0]
        df_dictionary["Fitness2"] = individual.fitness[1]
        df_dictionary["Fitness3"] = individual.fitness[2]

    df = pd.DataFrame.from_dict(df_dictionary)
    df.to_csv("pareto-front.csv", index=False)

    return

# calls the 'main' function when the script is called
if __name__ == "__main__" :
    sys.exit( main() )
