import json
import os
import shutil
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_data(file_path: str) -> list:
    """Parse list_of_data contained to the file specified from the given file path.
    Returns list of lists with user id, movie id and rating."""
    data = []
    with open(file_path) as f:
        for rec in f:
            data.append([int(c) for c in rec.split('\t')[:-1]])
    data.sort(key=lambda x: (x[0], x[1]))

    return data


def find_users_and_movies(data: list) -> tuple:
    """Calculate distinct user and movie id'rmse contained in the given list_of_data list.
    Returns a tuple with lists of the user and movie id'rmse."""
    return list(set([row[0] for row in data])), list(set([row[1] for row in data]))


def initialize_ratings_matrix(data: list) -> dict:
    users, _ = find_users_and_movies(data)
    matr = {u: [] for u in users}

    for row in data:
        matr[row[0]].append((row[1], row[2]))

    return matr


def initialize_population(user: list, movies: list, pop_size: int) -> list:
    """Initialize a list of size pop_size with random vectors of size item_size, where each cell contains an
    integer value from the range [1, 5]."""

    initial = [[[j, np.random.randint(1, 1+5)] for j in movies] for i in range(pop_size)]

    for it in initial:
        for row in user:
            ind = movies.index(row[0])
            it[ind][1] = row[1]

    return initial


def get_user_neighbourhood(id: int, matr: dict, size: int = 10) -> dict:
    """Get size-nearest neighbours of the user with the given id from the given matrix, where distance is specified
    from the Pearson correlation coefficient."""
    tmp = [[u, get_pearson_metric(matr[id], v)] for u, v in matr.items()]
    tmp.sort(key=lambda x: x[1][0], reverse=True)
    tmp.pop(0)

    tmp = [row for row in tmp if row[1][0] >= 1.6]
    tmp = [row for row in tmp if len(row[1][1]) >= 2]

    return {row[0]: matr[row[0]] for row in tmp[0:size]}


def evaluate(vec: list, neighbourhood: dict) -> float:
    """Calculate distance of the given ratings vector from the given neighbourhood, where distance is calculated using
    the sum of the Pearson correlation coefficients from each neighbour."""
    s = 0
    for n in neighbourhood.values():
        s += get_pearson_metric(vec, n)[0]
    return s/len(neighbourhood)


def eval_population(pop: list, neighbourhood: dict) -> float:
    """Evaluate the whole population from the given neighbourhood using the Pearson correlation coefficient. Returns the
    sum of the evaluation of each item of the population."""
    s = 0
    for p in pop:
        s += evaluate(p, neighbourhood)
    return s


def find_best(pop: list, neighbourhood: dict) -> tuple:
    """Get the index of the best item of the given population, which we measure evaluating all items from the given
    neighbourhood using the Pearson correlation coefficient."""
    best, r = (None, 0)
    for i in range(len(pop)):
        tmp = evaluate(pop[i], neighbourhood)
        if tmp > r:
            best = i
            r = tmp
    return best, r


def selection(pop: list, neighbourhood: dict, roulette: str = 'cost') -> list:
    """Select items that will be passed to the next generation using the method of roulette. Crossover and mutation
    are not done yet."""
    def cost_based_roulette_probs(evals):
        total_ev = eval_population(pop, neighbourhood)
        probs = [f / total_ev for f in evals]
        return probs

    def ranking_based_roulette_probs(evals):
        tmp = [(i, evals[i]) for i in range(len(evals))]
        tmp.sort(key=lambda x: x[1], reverse=True)
        c = 2
        probs = [0 for p in evals]
        for j, f in tmp:
            probs[j] = 1/c
            c += 1

        return probs

    # Calculate selection probability of every element
    evals = [evaluate(p, neighbourhood) for p in pop]
    if not (roulette == 'cost' or roulette == 'rank'):
        print('No such roulette option')
        exit(-1)
    probs = cost_based_roulette_probs(evals) if roulette == 'cost' else ranking_based_roulette_probs(evals)

    # Calculate accumulative probabilities -> construct roulette
    qs = [sum(probs[:i]) for i in range(1, 1+len(probs))]
    maxq = qs[-1]
    qs = [q/maxq for q in qs]

    # Select items from the population -> roll roulette len(pop) times
    res_pop = []
    for i in range(len(pop)):
        r = np.random.rand()
        for q in qs:
            if r <= q:
                res_pop.append(pop[qs.index(q)])
                break

    return res_pop


def crossover(pop: list, neighbourhood: dict, pc: float) -> list:
    """Change the genotypes of some items of the given population using the simple crossover method, where we create
    pairs of items using the given probability and then exchange parts of their genotypes."""
    # Choose items and create pairs. If the number of chosen items is odd then we ignore the last one that was chosen
    chosen = [i for i in range(len(pop)) if np.random.rand() <= pc]

    if len(chosen) % 2:
        chosen.pop(-1)
    pairs = [(chosen[i], chosen[i+1]) for i in range(0, len(chosen), 2)]

    # Crossover for each pair
    for a, b in pairs:
        pos = np.random.randint(1, len(pop[a]))
        tmp1 = [*pop[a][:pos], *pop[b][pos:]]
        tmp2 = [*pop[b][:pos], *pop[a][pos:]]
        pop[a], pop[b] = tmp1, tmp2

    return pop


def mutation(user: list, movies: list, pop: list, neighbourhood, pm: float) -> list:
    """Change the values of some digits of the whole population. The genotype of the item with the greatest
    evaluation will not be changed."""
    values = {1, 2, 3, 4, 5}
    best = find_best(pop, neighbourhood)

    user_movies = [movies.index(c[0]) for c in user]

    # Mutate items of the population except the item with the greatest evaluation value and except the values contained
    # at the initial user vector
    for i in list(set(range(len(pop))).difference({best})):
        indeces = list(set(range(len(pop))).difference(set(user_movies)))
        for j in indeces:
            if np.random.rand() <= pm:
                pop[i][j][1] = np.random.choice(list(values.difference({pop[i][j][1]})))

    return pop


def run_genetic_algorithm(user: list, movies: list, neighbourhood: dict, num_generations: int = 1000,
                          pop_size: int = 20, pc: float = 0.6, pm: float = 0.01, num_trials: int = 10,
                          leading_str: str = '') -> tuple:
    """Run the genetic algorithm with the given parameters. Return the population at the last step of the algorithm,
    the index of the best solution and the distance of the best solution specified by the Pearson correlation
    coefficient. The algorithm stops when the evaluation of the best item of the population stops getting better
    or gets better very slowly for a number of generations or the maximum number of generations is reached."""
    # Initialize population randomly.
    pop = initialize_population(user, movies, pop_size)
    #                            len(neighbourhood[np.random.choice(list(neighbourhood.keys()))]))
    best_item, best_r, generations, trials = -1, 0, 0, 1
    curr_best_item, curr_best_r = -1, 0
    best_items_per_generation = []
    performance_per_generation = []

    # Run the steps of the genetic algorithm -> genetic operations
    for i in range(num_generations):
        pop = mutation(user, movies, crossover(selection(pop, neighbourhood, 'rank'), neighbourhood, pc),
                       neighbourhood, pm)
        sum_r = eval_population(pop, neighbourhood)
        curr_best_item, curr_best_r = find_best(pop, neighbourhood)
        generations = i+1
        best_items_per_generation.append(pop[curr_best_item])
        performance_per_generation.append(curr_best_r)
        if curr_best_r - best_r < 0.01:
            trials += 1
        else:
            best_item, best_r = curr_best_item, curr_best_r
            trials = 1
        print('\r\t', leading_str, '\tGeneration:', str(generations)+'/'+str(num_generations),
              ('\t\t' if generations < 10 else '\t'), 'Avg Pearson r:', round(sum_r/len(pop), 4),
              '\t\tBest item\'s Pearson r:', round(curr_best_r, 4), '\tTrials:', trials, end='')
        if trials == num_trials:
            break
    print()

    return pop, curr_best_item, curr_best_r, generations, best_items_per_generation, performance_per_generation


def create_results_directory(directory_name: str) -> None:
    """Create an empty directory at the specified path, in which the results files will be stored."""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    else:
        folder = directory_name
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def calculate_avg(list_of_data: list) -> list:
    """For a given list of data calculate the avg per generation."""
    longest_row = max([len(row) for row in list_of_data])

    performance_sums = [0 for i in range(longest_row)]
    num_of_ratings_per_cell = [0 for i in range(longest_row)]

    for row in list_of_data:
        for i in range(len(row)):
            performance_sums[i] += row[i]
            num_of_ratings_per_cell[i] += 1

    avgs = [performance_sums[i]/num_of_ratings_per_cell[i] for i in range(longest_row)]

    return avgs


def store_results(all_results: list, results_directory) -> None:
    """Calculate wanted results and store them to JSON form and plots to .png files."""
    # Create the results directory
    create_results_directory(results_directory)

    # Create and store (performance/generation) plots
    full_performance = [[iteration['Performance per generation'] for iteration in res] for res in all_results]
    avg_performance = [calculate_avg(perf) for perf in full_performance]
    for i in range(len(avg_performance)):
        plt.plot(avg_performance[i])
        plt.ylabel('Performance')
        plt.xlabel('Generations')
        plt.savefig(results_directory+'/results'+str(i+1)+'_figure.png')
        plt.clf()

    # Create and store results JSONs
    results_json = {i+1: all_results[i] for i in range(len(all_results))}
    for k, v in results_json.items():
        filename = results_directory+'/results'+str(k)+'.json'
        with open(filename, 'w') as f:
            json.dump(v, f)


def rmse(x: list, y: list) -> float:
    """Count RMSE for the given vectors."""
    s = 0
    for i in range(len(x)):
        s += pow((x[i][1]-y[i][1]), 2)
    return sqrt(s/len(x))


def mae(x:list, y:list) -> float:
    """Count MAE for the given vectors."""
    s = 0
    for i in range(len(x)):
        s += abs(x[i][1]-y[i][1])
    return s/len(x)


def run_tests(user: list, movies: list, parameters: list, neighbourhood: dict,
              directory, num_of_iterations: int = 10) -> tuple:
    """Run the tests for the genetic algorithm and store results to the "./Results" directory. The results are stored
    in JSON form and at (generation, performance) plots for each of the various parameters of the problem."""
    test_counter = 0
    all_results = [[] for row in parameters]

    # Run tests for each parameter combination (population_size, crossover_probability, mutation probability)
    for pop_size, pc, pm in parameters:
        test_counter += 1

        # Run algorithm for given parameters 10 times
        print('Test', test_counter, ':\n\tPopulation size:', pop_size, ',\tCrossover probability:',
              pc, ',\tMutation Probability:', pm)
        for i in range(num_of_iterations):
            final_pop, best_item, best_r, num_generations, best_items_per_generation, performance_per_generation =\
                run_genetic_algorithm(user, movies, neighbourhood, pop_size=pop_size, pc=pc, pm=pm,
                                      leading_str=('\tIteration '+str(i+1)))

            iter_results = {'Population': [[(rating[0], int(rating[1])) for rating in p] for p in final_pop],
                            'Best item': best_item,
                            'Best r': best_r,
                            'Num of generations': num_generations,
                            'Best items per generation': [[(rating[0], int(rating[1])) for rating in row]
                                                          for row in best_items_per_generation],
                            'Performance per generation': performance_per_generation}
            all_results[test_counter-1].append(iter_results)

    store_results(all_results, directory)
    avg_best_rs = [round(sum([iteration['Best r'] for iteration in res]) / len(res), 4)
                   for res in all_results]
    avg_num_generations = [round(sum([iteration['Num of generations'] for iteration in res]) / len(res))
                           for res in all_results]
    print('\tResults:')
    for i in range(len(avg_best_rs)):
        print('\t\tTest', i+1, '\tAvg Pearson r:', avg_best_rs[i], '\tAvg num of generations:', avg_num_generations[i])

    return all_results, avg_best_rs, avg_num_generations


def calculate_and_export_rmses_maes(user_vector: list, results: list, avg_best_rs: list, movies: list,
                                    directory: str) -> tuple:
    """Calculate and export the RMSE and MAE plots of the given solution compared to the real values specified by the
    test set."""
    # Find best parameters combination
    best_results_index = avg_best_rs.index(max(avg_best_rs))
    best_results = results[best_results_index]

    best_items = [res['Best items per generation'] for res in best_results]
    movies_indeces = [movies.index(row[0]) for row in user_vector]
    best_items = [[[vector[i] for i in movies_indeces] for vector in row] for row in best_items]

    rmses = [[rmse(item, user_vector) for item in res] for res in best_items]
    maes = [[mae(item, user_vector) for item in res] for res in best_items]

    avg_rmses = calculate_avg(rmses)
    avg_maes = calculate_avg(maes)

    # Plot mses and maes
    plt.plot(avg_rmses)
    plt.xlabel('Generations')
    plt.ylabel('Avg RMSE')
    plt.savefig(directory+'/RMSE.png')
    plt.clf()

    plt.plot(avg_maes)
    plt.xlabel('Generations')
    plt.ylabel('Avg MAE')
    plt.savefig(directory+'/MAE.png')
    plt.clf()

    return avg_rmses, avg_maes


def calculate_solution_accuracy(user_movie_ratings: list, movies: list, best_item: list) -> None:
    """Calculate the ration of the number of ratings of the solution that are equal to the ratings of the test set."""
    indeces = [row[0] for row in user_movie_ratings]
    wanted_ratings = [best_item[movies.index(i)] for i in indeces]
    user_test_ratings = [row[1] for row in user_movie_ratings]
    print(pd.DataFrame([[user_test_ratings[i], wanted_ratings[i][1]] for i in range(len(user_test_ratings))]))

    success = 0
    for i in range(len(wanted_ratings)):
        if wanted_ratings[i][1] == user_test_ratings[i]:
            success += 1
    print('Success:', (success / len(wanted_ratings)) * 100, '%')


def run_tests_for_users(test_users: list, matr: dict, test_ratings: dict, parameters: tuple, movies: list,
                        directory: str) -> None:
    """Run tests for 50 users with the best combination of the parameters and calculate andd export the RMSE and MAE
    of the solutions compared to the ratings that are contained in the test set."""
    pop_size, pc, pm = parameters
    neighbourhoods = {u: get_user_neighbourhood(u, matr) for u in test_users}

    user_rmse_mae = {u: [] for u in test_users}
    for k, v in neighbourhoods.items():
        pop, curr_best_item, _, _, _, _ = \
            run_genetic_algorithm(matr[k], movies, v, pop_size=pop_size, pc=pc, pm=pm, leading_str=('User ' + str(k)))
        vec = pop[curr_best_item]
        vec = [vec[movies.index(i)] for i in [row[0] for row in test_ratings[k]]]

        user_rmse_mae[k] = [rmse(vec, test_ratings[k]),
                            mae(vec, test_ratings[k])]

    pd.DataFrame(user_rmse_mae).to_csv(directory+'/users_rmse_mae.csv')


def get_results(directory: str, user_id: int = 1, data_set_name: str = 'ua') -> None:
    """Run tests and get results."""
    training_data = parse_data('./ml-100k/'+data_set_name+'.base')
    test_data = parse_data('./ml-100k/'+data_set_name+'.test')
    users, movies = find_users_and_movies(training_data)
    matr = initialize_ratings_matrix(training_data)

    neighb = get_user_neighbourhood(user_id, matr)

    test_users = list(set([row[0] for row in test_data]))

    test_ratings = initialize_ratings_matrix(test_data)

    # Initialize the various test parameters
    parameters = [(20, 0.6, 0.00),
                  (20, 0.6, 0.01),
                  (20, 0.6, 0.10),
                  (20, 0.9, 0.01),
                  (20, 0.1, 0.01),
                  (200, 0.6, 0.00),
                  (200, 0.6, 0.01),
                  (200, 0.1, 0.01),
                  (200, 0.9, 0.01)]

    results, avg_best_rs, avg_num_generations = run_tests(matr[user_id], movies, parameters, neighb, directory)
    calculate_and_export_rmses_maes(test_ratings[user_id], results, avg_best_rs, movies, directory)

    best_results_index = avg_best_rs.index(max(avg_best_rs))
    best_results = results[best_results_index][-1]
    best_item = best_results['Population'][best_results['Best item']]
    calculate_solution_accuracy(test_ratings[user_id], movies, best_item)

    best_parameters = parameters[best_results_index]
    print('Best test')
    print('\tTest number:', best_results_index + 1)
    print('\tParameters:', *best_parameters, sep='\t')

    users_to_get_results = test_users[:50]
    run_tests_for_users(users_to_get_results, matr, test_ratings, best_parameters, movies, directory)


def get_pearson_metric(u1: list, u2: list) -> tuple:
    mov_ind_1 = {u1[i][0]: i for i in range(len(u1))}
    mov_ind_2 = {u2[i][0]: i for i in range(len(u2))}

    m1 = [k for k in mov_ind_1.keys()]
    m2 = [k for k in mov_ind_2.keys()]

    com = list(set(m1).intersection(set(m2)))
    com.sort()

    r1 = [u1[mov_ind_1[i]][1] for i in com]
    r2 = [u2[mov_ind_2[i]][1] for i in com]

    r1_avg = sum([c[1] for c in u1])/len(u1)
    r2_avg = sum([c[1] for c in u2])/len(u2)

    a = sum([(r1[i]-r1_avg)*(r2[i]-r2_avg) for i in range(len(r1))])
    b1 = sqrt(sum([pow(r1[i]-r1_avg, 2) for i in range(len(r1))]))
    b2 = sqrt(sum([pow(r2[i]-r2_avg, 2) for i in range(len(r2))]))

    return 1+(a/(b1*b2)) if (b1 and b2) else 0, com


if __name__ == '__main__':
    get_results('./Results_ua51', data_set_name='ua')
    get_results('./Results_ua52', data_set_name='ua')
    get_results('./Results_ub51', data_set_name='ub')
    get_results('./Results_ub52', data_set_name='ub')

