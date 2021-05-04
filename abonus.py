# abonus.py

# template for Bonus Assignment, Artificial Intelligence Survey, CMPT 310 D200
# Spring 2021, Simon Fraser University

# author: Jens Classen (jclassen@sfu.ca)

from learning import *


def generate_restaurant_dataset(size=100):
    """
    Generate a data set for the restaurant scenario, using a numerical
    representation that can be used for neural networks. Examples will
    be newly created at random from the "real" restaurant decision
    tree.
    :param size: number of examples to be included
    """

    numeric_examples = None

    ### YOUR CODE HERE ###

    non_numeric_examples = SyntheticRestaurant(size).examples
    # print_table(non_numeric_examples)

    # cols 0,1,2,3,6,7,10 are boolean attributes
    bool_attrs = [0, 1, 2, 3, 6, 7, 10]

    for i in non_numeric_examples:
        for j in bool_attrs:
            if i[j] == 'No':
                i[j] = 0
            elif i[j] == 'Yes':
                i[j] = 1

        # col 4 is the patrons attribute
        if i[4] == 'None':
            i[4] = 0
        elif i[4] == 'Some':
            i[4] = 1
        elif i[4] == 'Full':
            i[4] = 2

        # col 5 is the price attribute
        if i[5] == '$':
            i[5] = 0
        elif i[5] == '$$':
            i[5] = 1
        elif i[5] == '$$$':
            i[5] = 2

        # col 9 is the wait estimate attribute
        if i[9] == '0-10':
            i[9] = 0
        elif i[9] == '10-30':
            i[9] = 1
        elif i[9] == '30-60':
            i[9] = 2
        elif i[9] == '>60':
            i[9] = 3

    # col 8 is the type attribute
    # after applying distributed encoding
    # cols 8,9,10,11 are coresspond to Burger, French, Italian, Thai
    for i in non_numeric_examples:
        if i[8] == 'Burger':
            i[8] = 1
            i.insert(9, 0)
            i.insert(10, 0)
            i.insert(11, 0)
        elif i[8] == 'French':
            i[8] = 0
            i.insert(9, 1)
            i.insert(10, 0)
            i.insert(11, 0)
        elif i[8] == 'Italian':
            i[8] = 0
            i.insert(9, 0)
            i.insert(10, 1)
            i.insert(11, 0)
        elif i[8] == 'Thai':
            i[8] = 0
            i.insert(9, 0)
            i.insert(10, 0)
            i.insert(11, 1)

    # print(SyntheticRestaurant(size).attr_names)
    # print_table(non_numeric_examples)

    numeric_examples = non_numeric_examples

    return DataSet(name='restaurant_numeric',
                   target='Wait',
                   examples=numeric_examples,
                   attr_names='Alternate Bar Fri/Sat Hungry Patrons Price Raining Reservation Burger French Italian Thai WaitEstimate Wait')


def nn_cross_validation(dataset, hidden_units, epochs=100, k=10):
    """
    Perform k-fold cross-validation. In each round, train a
    feed-forward neural network with one hidden layer. Returns the
    error ratio averaged over all rounds.
    :param dataset:      the data set to be used
    :param hidden_units: the number of hidden units (one layer) of the neural nets to be created
    :param epochs:       the maximal number of epochs to be performed in a single round of training
    :param k:            k-parameter for cross-validation
                         (do k many rounds, use a different 1/k of data for testing in each round)
    """

    error = 0

    ### YOUR CODE HERE ###

    n = len(dataset.examples)
    # examples = dataset.examples
    random.shuffle(dataset.examples)

    for fold in range(k):
        train_example, val_example = train_test_split(
            dataset, fold * (n // k), (fold + 1) * (n // k))
        train_dataset = DataSet(name='restaurant_numeric',
                                target='Wait',
                                examples=train_example,
                                attr_names='Alternate Bar Fri/Sat Hungry Patrons Price Raining Reservation Burger French Italian Thai WaitEstimate Wait')
        predict = NeuralNetLearner(train_dataset, [hidden_units], 0.01, epochs)
        error += err_ratio(predict, dataset, val_example)

    return error/k


N          = 100   # number of examples to be used in experiments
k          =   5   # k parameter
epochs     = 100   # maximal number of epochs to be used in each training round
size_limit = 15   # maximal number of hidden units to be considered

# generate a new, random data set
# use the same data set for all following experiments
dataset = generate_restaurant_dataset(N)
# print(dataset.values)

# print(dataset.attr_names)
# print_table(dataset.examples)
# print(dataset.target)

# try out possible numbers of hidden units
for hidden_units in range(1, size_limit+1):
    # do cross-validation
    error = nn_cross_validation(dataset=dataset,
                                hidden_units=hidden_units,
                                epochs=epochs,
                                k=k)
    # report size and error ratio
    print("Size " + str(hidden_units) + ":", error)