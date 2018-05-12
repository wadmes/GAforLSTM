import pandas as pd
import lstm_model
import simulation
import random
import numpy as np
import csv

class conf:
    # Data
    instrument = 'AAPL'
    start_date = '2008-01-01'
    split_date1 = '2015-01-01'
    split_date2 = '2016-09-01'
    end_date = '2016-12-30'
    fields = ['change']  # features
    # Simulation
    upper_threshold = np.arange(0,0.6,0.1)
    lower_threshold = np.arange(-0.5,0.1,0.1)
    adjust_range = np.arange(0.05,0.55,0.05)
    adjust_ratio = np.arange(0.1,0.6,0.1)
    # LSTM
    seq_len = np.arange(1,20,1) #length of input
    layer_type = np.array(['GRU', 'LSTM', 'SimpleRNN'])
    layer_num = np.array([1,2,3])
    rnn_unit = np.array([16,32,64,128])
    dense_unit = np.array([32,64,128])
    batch = 128 # batch size
    epochs = 10 # num of epochs to train
    scale ='minmax' # way of scale training data, either minmax or standard
    # GP
    gene_num = 9
    generation_num = 1
    population_size = 2
    pc = 0.8 # Probability for crossover
    pr = 0.1 # Probability for reproduction
    pm = 0.1 # Probability for mutation
    # Genebase
    gene_base = []
    gene_base.append(upper_threshold) # 0
    gene_base.append(lower_threshold) # 1
    gene_base.append(adjust_range) # 2
    gene_base.append(adjust_ratio) # 3
    gene_base.append(seq_len) # 4
    gene_base.append(layer_type) # 5
    gene_base.append(layer_num) # 6
    gene_base.append(rnn_unit) # 7
    gene_base.append(dense_unit) # 8

def processData(data,label,lb):
    X,Y = [],[]
    for i in range(len(data)-lb):
        X.append(data[i:(i+lb)])
        Y.append(label[i+lb])
    return np.array(X), np.array(Y)

def fitness_measure(simulate_list, real_list):
    return_simulate = (simulate_list[-1] - simulate_list[0])/simulate_list[0]
    return_real = (real_list[-1] - real_list[0])/real_list[0]
    return return_simulate - return_real

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def select(fitness, num):
    return_list = []
    f = fitness
    remain_list = range(len(f))
    while len(return_list) < num:
        prob = softmax(f)
        accumulate_prob = 0
        rand = random.uniform(0,1)
        for i in range(prob.shape[0]):
            if rand >= accumulate_prob and rand < accumulate_prob + prob[i]:
                return_list.append(remain_list.pop(i))
                f.pop(i)
                break
            accumulate_prob += prob[i]
    return return_list

def crossover(father, mother):
    child1 = father
    child2 = mother
    for i in range(conf.gene_num):
        if random.uniform(0,1) < 0.4:
            x = child1[i]
            child1[i] = child2[i]
            child2[i] = x
    return child1, child2

def reproduction(parent):
    return parent

def mutation(parent):
    mut_ix = random.sample(range(conf.gene_num), 1)[0]
    current_ix = list(gene_base[mut_ix]).index(parent[mut_ix])
    if current_ix == 0:
        new_ix = current_ix + 1
    elif current_ix == gene_base[mut_ix].shape[0] - 1:
        new_ix = current_ix - 1
    else:
        if random.uniform(0,1) < 0.5:
            new_ix = current_ix + 1
        else:
            new_ix = current_ix - 1
    parent[mut_ix] = gene_base[mut_ix][new_ix]
    return parent

def write_to_csv(date, simulate, real, filename):
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Date','Simulate','Real'])
        for row in zip(date, simulate, real):
            writer.writerow(row)
    
def GP(gene_num=conf.gene_num, generation_num=conf.generation_num,
       population_size=conf.population_size,pc=conf.pc,pr=conf.pr,pm=conf.pm):
    # Load data
    print 'Load data...'
    data=pd.read_csv('../data/apple_'+conf.scale+'.csv')
    sentiment_raw=pd.read_csv('../data/sentiment_stat.csv')
    train = data[data.date<conf.split_date1]
    test = data[(data.date>=conf.split_date1) & (data.date<conf.split_date2)]
    test_date=test['date']
    train_date=train['date']
    sentiment_raw.set_index('Date', inplace=True)
    sentiment = sentiment_raw.loc[list(test_date)]

    # Initialize population
    print 'Initialize population...'
    population = [[None for _ in range(gene_num)] for _ in range(population_size)]
    for individual in population:
        for i in range(gene_num):
            individual[i] = random.sample(conf.gene_base[i],1)[0]

    # Start genetic programming
    print 'Start genetic programming...'
    for g in range(generation_num):
        print 'Generation {}...'.format(g)
        print 'Compute fitness...'
        fitness = []
        for individual in population: 
            test_x,test_y = processData(np.array(test[conf.fields]),np.array(test['next_change']),individual[4])
            train_x,train_y = processData(np.array(train[conf.fields]),np.array(train['next_change']),individual[4])
            model = lstm_model.generate_model(individual[4],individual[5],individual[6],individual[7],individual[8])
            model.fit(train_x,train_y,epochs=conf.epochs,batch_size=conf.batch,shuffle=False,validation_split=0.1)
            predictions = model.predict(test_x)
            capital_simulate, capital_stock = simulation.simulation_algorithm(
                sentiment, predictions, list(test['adj_close']), individual[0], individual[1],
                individual[2], individual[3])
            fitness.append(fitness_measure(capital_simulate, capital_stock))
        print 'Generate offspring...'
        next_generation = []
        while len(next_generation) < population_size:
            rand = random.uniform(0,1)
            if rand < pc: # Crossover
                ix = select(fitness, 2)
                father = population[ix[0]]
                mother = population[ix[1]]
                child1, child2 = crossover(father, mother)
                next_generation.append(child1)
                next_generation.append(child2)
            elif rand >= pc and rand < pc + pr: # Reproduction
                ix = select(fitness, 1)
                parent = population[ix[0]]
                child = reproduction(parent)
                next_generation.append(child)
            elif rand >= pc + pr and rand <= pc + pr + pm: # Mutation
                ix = select(fitness, 1)
                parent = population[ix[0]]
                child = mutation(parent)
                next_generation.append(child)
        population = next_generation

    # Select the best individual
    print 'Select the best individual...'
    fitness = []
    for individual in population:
        test_x,test_y = processData(np.array(test[conf.fields]),np.array(test['next_change']),individual[4])
        train_x,train_y = processData(np.array(train[conf.fields]),np.array(train['next_change']),individual[4])
        model = lstm_model.generate_model(individual[4],individual[5],individual[6],individual[7],individual[8])
        model.fit(train_x,train_y,epochs=conf.epochs,batch_size=conf.batch,shuffle=False,validation_split=0.1)
        predictions = model.predict(test_x)
        capital_simulate, capital_stock = simulation.simulation_algorithm(
            sentiment, predictions, list(test['adj_close']), individual[0], individual[1],
            individual[2], individual[3])
        fitness.append(fitness_measure(capital_simulate, capital_stock))
    best_individual = population[fitness.index(max(fitness))]
    print 'The best individual: {}'.format(best_individual)

    # Evaluate the best individual (Compare with baseline models)
    train = data[data.date<conf.split_date2]
    test = data[(data.date>=conf.split_date2) & (data.date<=conf.end_date)]
    test_date=test['date']
    sentiment = sentiment_raw.loc[list(test_date)]
    # Best individual
    print 'Evaluate the best individual...'
    test_x,test_y = processData(np.array(test[conf.fields]),np.array(test['next_change']),best_individual[4])
    train_x,train_y = processData(np.array(train[conf.fields]),np.array(train['next_change']),best_individual[4])
    model = lstm_model.generate_model(best_individual[4],best_individual[5],best_individual[6],best_individual[7],best_individual[8])
    model.fit(train_x,train_y,epochs=conf.epochs,batch_size=conf.batch,shuffle=False,validation_split=0.1)
    predictions = model.predict(test_x)
    capital_simulate, capital_stock = simulation.simulation_algorithm(
        sentiment, predictions, list(test['adj_close']), best_individual[0], best_individual[1],
        best_individual[2], best_individual[3])
    GP_fitness = fitness_measure(capital_simulate, capital_stock)
    print 'GP_fitness: {}'.format(GP_fitness)
    with open('best_individual.txt','w') as file:
        file.write('GP_fitness: {}'.format(GP_fitness))
    print 'Write simulation process into csv file...'
    write_to_csv(list(test_date)[best_individual[4]:], capital_simulate, capital_stock, 'best_individual.csv')
    # Average individual
    print 'Evaluate average individual...'
    avg_individual = [None for _ in range(gene_num)]
    for i in range(gene_num):
        avg_individual[i] = conf.gene_base[i][conf.gene_base[i].shape[0]/2]
    test_x,test_y = processData(np.array(test[conf.fields]),np.array(test['next_change']),avg_individual[4])
    train_x,train_y = processData(np.array(train[conf.fields]),np.array(train['next_change']),avg_individual[4])
    model = lstm_model.generate_model(avg_individual[4],avg_individual[5],avg_individual[6],avg_individual[7],avg_individual[8])
    model.fit(train_x,train_y,epochs=conf.epochs,batch_size=conf.batch,shuffle=False,validation_split=0.1)
    predictions = model.predict(test_x)
    capital_simulate, capital_stock = simulation.simulation_algorithm(
        sentiment, predictions, list(test['adj_close']), avg_individual[0], avg_individual[1],
        avg_individual[2], avg_individual[3])
    avg_fitness = fitness_measure(capital_simulate, capital_stock)
    print 'avg_fitness: {}'.format(avg_fitness)
    with open('avg_individual.txt','w') as file:
        file.write('avg_fitness: {}'.format(avg_fitness))
    print 'Write simulation process into csv file...'
    write_to_csv(list(test_date)[avg_individual[4]:], capital_simulate, capital_stock, 'avg_individual.csv')
    # Random guess
    capital_simulate, capital_stock = simulation.random_guess(list(test['adj_close']))
    random_guess_fitness = fitness_measure(capital_simulate, capital_stock)
    print 'random_guess_fitness: {}'.format(random_guess_fitness)
    with open('random_guess.txt','w') as file:
        file.write('random_guess_fitness: {}'.format(random_guess_fitness))
    print 'Write simulation process into csv file...'
    write_to_csv(list(test_date)[1:], capital_simulate, capital_stock, 'random_guess.csv')

if __name__ == '__main__':
    GP()
