import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg


# returns 12 feature weights np array based on a penalty regression factor
def ridge_regression(features_train, y_train, regularisation_factor):

    identity_ = np.identity(features_train.shape[1])
    parameters = linalg.solve((features_train.transpose().dot(features_train) +
                               (regularisation_factor * identity_)),
                              features_train.transpose().dot(y_train))

    return np.array(parameters)


# calculates the error between the predicted y and actual y
# returns 12 predicted y for each 12 features in training
def rr_squared_error(weights, x, y_train):

    prediction_rr = []
    error_rr = 0

    for ys in range(12):
        y_rr = 0  # rr_weights0[0]

        for i in range(12):
            y_rr = y_rr + (weights[i] * x[ys][i])

        prediction_rr.append(y_rr)

        error_rr = error_rr + ((y_train[ys] - y_rr) ** 2)

        print("                 Actual Y: " + str(y_train[ys]))
        print(" Predicted RR; 0.000001 Y: " + str(y_rr))
        print("                 Error RR: " + str(abs((y_train[ys] - y_rr))))
        print("_________________________________________________________________________________")

    print("RR, Residual Sum of Squares (RSS): " + str(error_rr))

    return prediction_rr


# returns 100 predicted y values for each 12 features in testing data
def prediction_y_rr_reg(weights, plotting_x):

    prediction_y_plotting = []

    for ii in range(100):
        y_plot = 0  # rr_weights0[0]

        for i in range(12):
            y_plot = y_plot + (weights[i] * plotting_x[ii][i])

        prediction_y_plotting.append(y_plot)

    print(len(prediction_y_plotting))
    return prediction_y_plotting


# plots the 4 regularisation factors predicted y values for training against test X
# plots the 4 regularisation factors predicted y values for testing (plotting) against test X
def plotting_predicted_y(x_train, x_plot, rr_prediction__y, prediction_plot):

    # displays regularisation 0.000001 predicted y values for training and testing against x for both
    plt.figure()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("0.000001")
    plt.ylim(-1000, 1000)
    plt.plot(x_train, rr_prediction__y[0], 'bo', label='Training Predict')
    plt.plot(x_plot, prediction_plot[0], 'r', label='0.000001 Predict')

    # displays regularisation 0.0001 predicted y values for training and testing against x for both
    plt.figure()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("0.0001")
    plt.ylim(-1000, 1000)
    plt.plot(x_train, rr_prediction__y[1], 'bo', label='Training Predict')
    plt.plot(x_plot, prediction_plot[1], 'r', label='0.0001 Predict')

    # displays regularisation 0.01 predicted y values for training and testing against x for both
    plt.figure()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("0.01")
    plt.ylim(-1000, 1000)
    plt.plot(x_train, rr_prediction__y[2], 'bo', label='Training Predict')
    plt.plot(x_plot, prediction_plot[2], 'r', label='0.01 Predict')

    # displays regularisation 0.1 predicted y values for training and testing against x for both
    plt.figure()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("0.1")
    plt.ylim(-1000, 1000)
    plt.plot(x_train, rr_prediction__y[3], 'bo', label='Training Predict')
    plt.plot(x_plot, prediction_plot[3], 'r', label='0.1 Predict')
    plt.show()


# produces the RMSE (root mean squared error)
def eval_regression(parameters, features, y):
    # produces a 1 d array
    predy = features.dot(parameters)
    # calculates the different between actual and prediction
    difference = y - predy
    # squares the error (creates a positive)
    difference = difference**2
    # sums the differences into var
    totalsum = sum(difference)
    # divides by the length of y -1
    rmse2 = totalsum / (len(y)-1)
    # square root the answer and return single variable
    rmse = np.sqrt(rmse2)

    return rmse


# plots the x1 of 70% train and 30% test data against all 8 reg factors RSME
# plots the x10 of random 70% train and 30% test against all 8 reg factors RSME
def plotting_1x_10x_reg_factors(holder_rsme_train_, holder_rsme_test_, reg_plot_train, reg_plot_test, regularisation_factors):

    print(regularisation_factors)
    print(holder_rsme_test_)

    plt.figure()
    plt.xlabel("Reg Factors")
    plt.ylabel("Y")
    plt.title("RSME's 1X")
    plt.plot(regularisation_factors, holder_rsme_train_, 'b')
    plt.plot(regularisation_factors, holder_rsme_test_, 'r')
    plt.yscale('log')
    plt.xscale('log')

    plt.figure()
    plt.xlabel("Reg Factors")
    plt.ylabel("Y")
    plt.title("RSME's 10X")
    plt.plot(regularisation_factors, reg_plot_train, 'b')
    plt.plot(regularisation_factors, reg_plot_test, 'r')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()


# train and test RSME calculation
# returns 2 by 8 list of train and test for each 8 regularisation factors
def single_reg_factor_testing(reg_factors, random):

    # reads in the training file
    data_train = pd.read_csv('regression_train_assignment2019.csv')

    # checks if doing a random sample
    if random:

        # randomises the data to select testing and training data
        data_train = data_train.sample(frac=1).reset_index(drop=True)

    # Preprocessing Data
    # takes the y column, convert to np.array, and drope the x and y columns
    # deletes the first row to remove index
    y_plot = data_train['y']
    y_plot = np.array(y_plot)
    data_train.drop(['x', 'y'], axis=1, inplace=True)
    train = data_train.values
    train = train.transpose()
    train = np.array(train)
    train = np.delete(train, 0, axis=0)
    train = train.transpose()

    # splits the data in to 8 train and 4 test
    split = 8
    training, test = train[:split, :], train[split:, :]

    # creates an identity matrix to the size of training first column
    identity_ = np.identity(training.shape[1])

    # stores the weights for all regularisations
    # output 4x12
    holder_rr_weights = []
    for reg in range(8):
        holder_rr_weights.append(
            linalg.solve((training.transpose().dot(training) + reg_factors[reg] * identity_),
                         training.transpose().dot(y_plot[0:8])))

    # stores all the training and testing for RSME values for each reg factors
    # outputs to lists of RSME for each regularisation
    holder_rsme_train = []
    holder_rsme_test = []

    for test_RSME in range(8):
        holder_rsme_train.append(eval_regression(holder_rr_weights[test_RSME], training, y_plot[0:8]))
        holder_rsme_test.append(eval_regression(holder_rr_weights[test_RSME], test, y_plot[8:12]))

    return holder_rsme_train, holder_rsme_test


# solution for section 1.2
def implementation_ridge_regression(regularisation_factors):
    # read in files training & plotting
    data_train = pd.read_csv('regression_train_assignment2019.csv')
    data_plot = pd.read_csv('regression_plotting_assignment2019.csv')

    # copies the training data x and y values to a separate entity
    x_train = data_train['x']
    y_train = data_train['y']

    # convert y_train to np array
    y = np.array(y_train)

    # copy the data_plot file x values to a separate entity
    x_plot = data_plot['x']

    # drop the x and y columns from the training data
    # take all the training data values minus headings and place them in a np array
    data_train.drop(['x', 'y'], axis=1, inplace=True)
    data_train = data_train.values
    data_train = data_train.transpose()
    features = np.array(data_train)

    # drop the x and y columns from the testing (plotting) data
    # take all the training data values minus headings and place them in a np array
    data_plot.drop(['x'], axis=1, inplace=True)
    data_plot = data_plot.values
    data_plot = data_plot.transpose()
    features_plot = np.array(data_plot)

    # delete the first row of the features (training) and the features_plot (testing)
    features = np.delete(features, (0), axis=0)
    features_train = features.transpose()
    features_plot = np.delete(features_plot, (0), axis=0)
    features_plot = features_plot.transpose()

    # calculates the 12 weighting for each feature with the regularisation
    # output 4x12
    rr_weights_e_6_1 = []
    for reg_i in range(4):
        rr_weights_e_6_1.append(ridge_regression(features_train, y, regularisation_factors[reg_i]))

    # calculates the features_trains y values using the weights for each 4 regularisation factors
    rr_prediction__y = []
    for reg_i in range(4):
        rr_prediction__y.append(rr_squared_error(rr_weights_e_6_1[reg_i], features_train, y))

    # calculates the features_plot y values using the weights for each 4 regularisation factors
    rr_prediction_plotting_y = []
    for reg_i in range(4):
        rr_prediction_plotting_y.append(prediction_y_rr_reg(rr_weights_e_6_1[reg_i], features_plot))

    # paces over the x axis for train and plot along with the predicted values for each regularisation factor
    plotting_predicted_y(x_train, x_plot, rr_prediction__y, rr_prediction_plotting_y)


# solution for section 1.3
def evaluation(regularisation_factors):

    # holds the RSME values for training and testing 70% 30%
    rsme_train_single, rsme_test_single = single_reg_factor_testing(regularisation_factors, False)

    # lists to store 10x randomisation of the training data set
    rsme_train_multiple = []
    rsme_test_multiple = []

    # runs through 10 factor testing with random data set 70% 30%
    for loop in range(10):

        # holds the RSME values for training and testing 70% 30%
        rsme_train_s, rsme_test_s = single_reg_factor_testing(regularisation_factors, True)

        # adds each testing to a list which stores 10
        rsme_train_multiple.append(rsme_train_s)
        rsme_test_multiple.append(rsme_test_s)

    # used for storing all 10x train and test rsme
    avg_reg_plot_train = []
    avg_reg_plot_test = []


    for it1 in range(8):
        train = 0
        test = 0

        # iterates through the all the iterates and creates and avg rsme
        for it2 in range(10):
            train = train + rsme_train_multiple[it2][it1]
            test = test + rsme_test_multiple[it2][it1]
        avg_reg_plot_train.append(train / 10)
        avg_reg_plot_test.append(test / 10)

    plotting_1x_10x_reg_factors(rsme_train_single, rsme_test_single, avg_reg_plot_train, avg_reg_plot_test, regularisation_factors)


# START OF MAIN

# Regularisation factors used in model and evaluation
regularisation_factors = [0.000001, 0.0001, 0.01, 0.1, 1, 10, 100, 1000]

# section 1.2 ridge regression
implementation_ridge_regression(regularisation_factors)

# section 1.3 evaluation
evaluation(regularisation_factors)
