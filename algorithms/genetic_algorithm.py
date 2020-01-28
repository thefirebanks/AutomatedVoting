""" Basic implementation of the genetic algorithm for the automated voting framework.
    Tutorial by: https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6

    Usage specified in genetic_main()

    TODO FEATURES:
        - In fitness function, we should be able to specify the parameters of a profile/election
        - Add the rest of constraints
        - Add more options for crossover if possible
        - Deal with multiple indices/repetitions in selecting mating pool
        - """

import numpy as np
from election import election
from constraints import check_condorcet, check_majority
from profiles import create_profile

''' Fairness Constraints:

    CC = Condorcet complicity 
    MR = Majority rule
    MT = Monotonicity
    IIA = Independence of Irrelevant Alternatives 

    Non-manipulability Constraints:

    IM = Individual manipulation
    CM = Coalition manipulation
    TM = Trivial manipulation
    UM = Unison manipulation

    Other constraints:

    PC = Population Consistency
    CS = Composition Consistency '''


def initialize_pop(low, high, num_solutions, num_weights):
    """ A population will be a set of weight vectors
    - num_solutions is the number of weight vectors
    - num_weights is the number of weights
    - low and high are the bounds for the weight values """

    pop_size = (num_solutions, num_weights)
    new_pop = np.random.uniform(low=low, high=high, size=pop_size)
    return new_pop


def crossover(parents, size, point="single"):
    """ Generates offspring from the parents using point (single by default) crossover """

    offspring = np.empty(size)

    # TODO: Afterwards add more options
    if point == "single":
        # The point in the chromosome after which crossover occurs (if single_point, usually half)
        crossover_point = np.uint8(size[1] / 2)

        # For every new weight vector
        for i in range(size[0]):
            # Pick the ith and ith+1 parents
            parent1_idx = i % parents.shape[0]
            parent2_idx = (i + 1) % parents.shape[0]

            # For ith offspring, first half of genes from first parent, second half from second
            offspring[i, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            offspring[i, crossover_point:] = parents[parent2_idx, crossover_point:]

    return offspring


def mutation(offspring_crossover, mutation_idx, low, high):
    """ Change one gene at mutation_idx from a given offspring randomly
        Value for mutation is in the range (low, high, 1) """

    for i in range(offspring_crossover.shape[0]):
        random_val = np.random.uniform(low, high, 1)
        offspring_crossover[i: mutation_idx] += random_val

    return offspring_crossover


def fitness(pop):  # TODO: Add later (fairness, non_manipulablity, others) as args -> or maybe **kwargs?
    """ Evaluate the fitness of the weight population by running an election
        and a test of the results based on the input fairness and non-manipulability constraints """

    # Profile for election
    profile, profile_df, profile_matrix = create_profile(500, origin="distribution",
                                                         params="spheroid", candidates=["Adam", "Bert",
                                                                                        "Chad", "Derek", "Elon"])
    fitness_pop = []

    for weights in pop:
        results = election(profile, weights)
        #     for result, value in results.items():
        #         print(f"{result}: {value}")

        # Score = condorcet compliance + majority rule
        condorcet_score = check_condorcet(profile, results)
        majority_score = check_majority(profile_df, results)

        fitness_pop.append(condorcet_score + majority_score)

    #     print("Is Condorcet compliant?:", check_condorcet(profile, results))
    #     print("Satisfies majority criterion?", check_majority(profile_df, results))

    return np.array(fitness_pop)


def select_mating_pool(pop, pop_fitness, n_parents):
    """ Select the best n_parents among the population according to their pop_fitness """

    parents = np.empty((n_parents, pop.shape[1]))  # (n_parents, n_weights)

    # For every possible parent that we could to take
    for parent_i in range(n_parents):
        # Get the ith best fitness parent index
        ##### TODO: What happens if multiple people have that index????
        max_fitness_idx = np.where(pop_fitness == np.max(pop_fitness))[0][0]

        # Get the parent with that index from pop and set it in the parents array
        parents[parent_i, :] = pop[max_fitness_idx, :]

        # Make sure that parent is not picked again
        #### ALSO TODO: What happens if there are repetitions?
        pop_fitness[max_fitness_idx] = -999999999

    return parents


def genetic_main():
    low, high = -10, 10
    n_sols, n_weights = 10, 3
    n_generations = 10
    n_parents = 4

    pop = initialize_pop(low, high, n_sols, n_weights)

    for generation in range(n_generations):
        # Measure the fitness of each chromosome in the population
        pop_fitness = fitness(pop)

        print("Generation:", generation)
        print("Weights:", pop)
        print("Fitness:", pop_fitness)
        print("-------------------------")

        # Select the best parents in the population for mating
        parents = select_mating_pool(pop, pop_fitness, n_parents)

        # Create next generation with crossover
        offspring_crossover = crossover(parents, size=(n_sols - parents.shape[0], n_weights))

        # Add mutations
        offspring_mutation = mutation(offspring_crossover, 3, -25, 25)

        # Create next population with new parents and offspring
        pop[0:parents.shape[0], :] = parents  # Set parents
        pop[parents.shape[0]:, :] = offspring_mutation  # Set offspring


if __name__ == "__main__":
    genetic_main()
