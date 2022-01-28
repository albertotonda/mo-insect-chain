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
from random import randrange

def load_instance(json_file):
    """
    Inputs: path to json file
    Outputs: json file object if it exists, or else returns NoneType
    """
    if os.path.exists(path=json_file):
        with io.open(json_file, 'rt', newline='') as file_object:
            return load(file_object)
        
    print("Error: cannot read file %s" % json_file)
    return None

# this function just converts an individual from the internal representation to the external representation
def convert_individual_to_values(candidate, boundaries) :

    SC = candidate["SC"]
    Nl = int(candidate["Nl"] * (boundaries["Nl"][SC][1] - boundaries["Nl"][SC][0]) + boundaries["Nl"][SC][0])
    AIF = candidate["AIF"] * (boundaries["AIF"][SC][1] - boundaries["AIF"][SC][0]) + boundaries["Nl"][SC][0]
    F = candidate["F"]
    EQ = candidate["EQ"]
    RW = candidate["RW"] * (boundaries["RW"][1] - boundaries["RW"][0]) + boundaries["RW"][0]
    

    return SC, Nl, AIF, F, EQ, RW

def fitness_function(candidate, json_instance, boundaries) :   

    operating_profit = 0 # maximize
    insect_frass = 0 # minimize
    labor = 0 # maximize
    labor_safety = 0 # maximize

    SC, Nl, AIF, F, EQ, RW = convert_individual_to_values(candidate, boundaries)

    
    # operating profit and frass
    biomass = 0.0
    feed_cost = 0.0
    insect_frass = 0.0
    labor_safety = 0
    
    # equipment cost
    equipment_cost = 0.0
    #weight = random.uniform(0, 1) # TODO why was this weight randomized?
    weight = 0.5
    
    for index, equip_dict in enumerate(json_instance["equipments"]) : 
        labor_safety += equip_dict["equipment_cost"] * EQ[index] * json_instance["SFls"][SC-1]/boundaries["EQ"] * Nl
    
    FWP = (RW / json_instance["RWT"])* (json_instance["CWT"]/json_instance["MLW"])*(1 - np.square(json_instance["IEF"]))
    social_aspect = weight * labor_safety + (1-weight) * FWP
        
    for index, feed_dict in enumerate(json_instance["feed"]) :
        biomass += AIF * F[index] * feed_dict["FCE"]
        feed_cost += AIF * F[index] * feed_dict["feed_cost"]
        insect_frass += AIF / feed_dict["FCE"] * (1.0 - feed_dict["FCE"]) * json_instance["Frsf"][SC-1]
        
    operating_profit = (json_instance["sales_price"]-json_instance["energy_cost"]) * biomass - RW*Nl*12 -json_instance["rent"][SC-1]-labor_safety - feed_cost
    
    print('------------------------')
    print(operating_profit)
    print(insect_frass)
    print(social_aspect)
    print('------------------------')
   

    return 1/operating_profit, 1/insect_frass, social_aspect


def generator(random, args) :

    boundaries = args["boundaries"]
    
    # here we need to generate a random individual and check that the boundaries are respected
    # to (hopefully) make our lives easier, individuals are encoded as dictionaries
    print("Generating new individual...")
    individual = dict()

    # first step: randomize the scale of the company end the real wages
    individual["SC"] = random.choice(boundaries["SC"])

    # other values are in (0,1), and will then be scaled depending on the scale of the company (before evaluation)
    individual["AIF"] = random.uniform(0, 1)
    individual["Nl"] = random.uniform(0, 1)
    individual["RW"] = random.uniform(0, 1)

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
    boundaries = args["boundaries"]

    list_of_fitness_values = []
    for candidate in candidates :
        f1, f2, f3 = fitness_function(candidate, json_instance, boundaries)
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
            number_of_bit_flips = min(random.randint(1, len(individual["EQ"])) for i in range(0, len(individual["EQ"])))
            # choose several equipments (with replacement)
            indexes = random.sample(range(0, len(individual["EQ"])), number_of_bit_flips)

            for index in indexes :
                if individual["EQ"][index] == 0 :
                    individual["EQ"][index] = 1
                else :
                    individual["EQ"][index] = 0

        elif to_be_mutated == "F" :
            # perform a random number of value modifications; low (high probability) or high (low probability)
            number_of_modifications = min(random.randint(1, len(individual["F"])) for i in range(0, len(individual["F"])))
            # choose several types of feed (with replacement)
            print("Number of modifications: %d" % number_of_modifications)
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
    json_instance = load_instance('../data/data.json') 

    # boundaries for all the values included in the individual
    boundaries = dict()
    boundaries["SC"] = [1, 2, 3, 4] # minimum and maximum
    boundaries["EQ"] = 5 # number of different types of equipments
    boundaries["F"] = 4 # types of different feeds
    boundaries["RW"] = [18096, 32268]

    # boundaries for AIF and Nl, depending on SC
    boundaries["AIF"] = dict()
    boundaries["AIF"][1] = [25000, 75000]
    boundaries["AIF"][2] = [75000, 125000]
    boundaries["AIF"][3] = [125000, 175000]
    boundaries["AIF"][4] = [175000, 250000]    
    
    boundaries["Nl"] = dict()
    boundaries["Nl"][1] = [25, 75]
    boundaries["Nl"][2] = [75, 125]
    boundaries["Nl"][3] = [125, 175]
    boundaries["Nl"][4] = [175, 250]
    
    # initialize random number generator
    random_number_generator = random.Random()
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
                            maximize = True, # TODO currently, it's trying to maximize EVERYTHING, so we need to 
                                             # have the fitness function output values that are better than higher
                                             # for example, use 1.0/value if the initial idea was to minimize

                            # all arguments specified below, THAT ARE NOT part of the "evolve" method, will be automatically placed in "args"
                            # "args" is a dictionary that is passed to all functions
                            boundaries = boundaries,
                            json_instance = json_instance,
    )

    # save the final Pareto front in a .csv file
    # prepare dictionary that will be later converted to .csv file using Pandas library
    df_dictionary = { "SC": [], "AIF": [], "Nl": [], "RW": [], "Economic_Impact": [], "Environmental_Impact": [],"Social_Impact": []} 
    for e in range(0, boundaries["EQ"]) :
        df_dictionary["EQ" + str(e)] = []
    for f in range(0, boundaries["F"]) :
        df_dictionary["F" + str(f)] = []

    # TODO change names of the fitnesses to their appropriate correspondence (e.g. "Profit", "Social Impact", "Environmental Impact")
    #df_dictionary["Economic_Impact"] = []
    #df_dictionary["Environmental_Impact"] = []
    #df_dictionary["Social_Impact"] = []

    # go over the list of individuals in the Pareto front and store them in the dictionary 
    # after converting them from the internal 'genome' representation to actual values
    for individual in final_pareto_front :

        #genome = individual.genome # uncomment this line and comment the two lines below to have the individuals saved with their internal representation
        SC, Nl, AIF, F, EQ, RW  = convert_individual_to_values(individual.candidate, boundaries)
        val_1= individual.fitness[0]
        val_2= individual.fitness[1]
        val_3= individual.fitness[2]
        genome = { "SC": SC, "Nl": Nl, "AIF": AIF, "F": F, "EQ": EQ, "RW": RW, "Economic_Impact": val_1, "Environmental_Impact": val_2, "Social_Impact": val_3}
        for k in genome :
            # manage parts of the genome who are lists
            if isinstance(genome[k], list) :
                for i in range(0, len(genome[k])) :
                    df_dictionary[k + str(i)].append(genome[k][i])
            else :
                df_dictionary[k].append(genome[k]) 
                 
        df = pd.DataFrame.from_dict(df_dictionary)
    df.to_csv("pareto-front.csv", index=False)

    return

# calls the 'main' function when the script is called
if __name__ == "__main__" :
    sys.exit( main() )
