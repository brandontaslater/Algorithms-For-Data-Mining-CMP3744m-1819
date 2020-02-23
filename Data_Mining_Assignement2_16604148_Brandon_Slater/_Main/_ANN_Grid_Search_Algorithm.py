import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# balances the data set
def balanced_normal_abnormal(FILE_PATH):
    # read the dataset into a np array, transposes
    data_set = (np.loadtxt(FILE_PATH, delimiter=",")).transpose()
    # extracts the normal statuses
    normal = (data_set[:, [i for i in range(498)]]).transpose()
    # extracts the abnormal statuses
    abnormal = (data_set[:, [i+498 for i in range(498)]]).transpose()
    # shuffles both the classes in the same way
    np.random.seed(0), np.random.shuffle(normal)
    np.random.seed(0), np.random.shuffle(abnormal)
    # creates a new array to store the balanced data
    k_fold = np.zeros((996, 13))
    count = 0  # used for new array
    step = 0  # used for looping the class arrays
    # loops through to the number of samples
    while count < 996:
        # the class arrays pass one each time loop into new balances array.
        k_fold[count] = normal[step]
        k_fold[count+1] = abnormal[step]
        count += 2  # increments count
        step += 1  # increments step
    # returns the balanced data set
    return k_fold


# function for model creation, number of neurons in the hidden layer| optimiser used with the hidden layer
def create_model(nodes=100, optimizer='adam'):
    CLASSIFICATIONS = 1  # number of reactor classes
    LOSS_FUNCTION = 'binary_crossentropy'  # used for categorical classes 2 or more
    HIDDEN_LAYER_ACTIVATION = 'sigmoid'  #
    OUTPUT_LAYER_ACTIVATION = 'sigmoid'  # logistic function
    INPUT_DIMENSION = 12  # number of features input

    # creating model
    model = Sequential()
    # input data into hidden layer
    model.add(Dense(nodes, input_dim=INPUT_DIMENSION, activation=HIDDEN_LAYER_ACTIVATION))
    # output classification layer using logistic activation function
    model.add(Dense(CLASSIFICATIONS, activation=OUTPUT_LAYER_ACTIVATION))

    # compile model
    model.compile(loss=LOSS_FUNCTION, optimizer=optimizer, metrics=['accuracy'])
    return model


# ------------- BEST PARAMETERS -------------
# Best: 0.898437 using {'batch_size': 256, 'epochs': 2000, 'nodes': 500}

# ------------- BEST PARAMETERS for 100 Neurons -------------
# Best: 0.893973 using {'batch_size': 512, 'epochs': 2000, 'optimizer': 'adam'}

# ------------- BEST PARAMETERS for 100 Neurons and Min-Max Scaling -------------
# Best: 0.726562 using {'batch_size': 256, 'epochs': 1000, 'optimizer': 'adam'}
# read the data set into a np array
DATA_SET = balanced_normal_abnormal('Data.csv')

# split data set into sets for testing and training
X = DATA_SET[:, 1:13]  # takes all the 12 features
Y = DATA_SET[:, 0:1]  # takes the label column
# splits the features and labels into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, shuffle=False)

# normalised the data through min-max scaling
standard_scaler = MinMaxScaler(feature_range=(0, 1))
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.fit_transform(x_test)

# creates model
model = KerasClassifier(build_fn=create_model, verbose=0)
# assigns different parameters for the grid search
epochs = np.array([250, 500, 1000, 2000])
nodes = np.array([100])
batches = np.array([128, 256, 512])
optimizer = np.array(['adam', 'RMSprop', 'SGD'])
# assigneds the above parameters to a gride search dict
param_grid = dict(epochs=epochs, batch_size=batches, optimizer=optimizer)
# applies the grid search based on setup parameters
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=4)
grid_result = grid.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

