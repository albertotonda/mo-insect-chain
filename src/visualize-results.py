# Script to visualize the results of a multi-objective optimization (Pareto front)

import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys

sns.set_theme() # set the SeaBorn graphical theme, it looks nice 

def plot_2d(df, fitness_a, fitness_b) :

    # set the data
    fitness_a_values = df[fitness_a].values
    fitness_b_values = df[fitness_b].values

    # plot the figure
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.scatter(fitness_a_values, fitness_b_values, color='blue', alpha=0.7) # alpha sets transparency

    ax.set_title("Pareto front, %s vs %s" % (fitness_a, fitness_b))
    ax.set_xlabel(fitness_a)
    ax.set_ylabel(fitness_b)

    plt.savefig(os.path.join(output_folder, "pareto-front-" + fitness_a + "-" + fitness_b + ".png"), dpi=300)
    plt.close(fig)

    return

def main() :

    # a few hard-coded values
    input_file = "pareto-front.csv"
    fitness_names = ["Fitness1", "Fitness2", "Fitness3"] # these should be changed with the names of the corresponding columns in the CSV file
    # output folder with a unique name, using the current date and time
    output_folder = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+ "-" + sys.argv[0][:-3] 

    # read the CSV file
    print("Reading file \"%s\"..." % input_file)
    df = pd.read_csv(input_file)
    print(df)

    # create an output folder, if it does not already exist
    if not os.path.exists(output_folder) : os.makedirs(output_folder)

    # let's create some visualizations!
    plot_2d(df, fitness_names[0], fitness_names[1])
    plot_2d(df, fitness_names[1], fitness_names[2])
    plot_2d(df, fitness_names[0], fitness_names[2])

    # TODO also, a 3d plot with everything

    return

if __name__ == "__main__" :
    sys.exit( main() )
