# imports:
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


# setting the pd data frame to work within pycharm console interface
desired_width = 500
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 13)


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


# ====================================================================
# ========================== READ IN DATA ============================
# ====================================================================
# read the dataset into a np array
DATA_SET = balanced_normal_abnormal('Data.csv')

# ====================================================================
# ============================= Model ================================
# ====================================================================
TEST_SIZE = 0.10  # the size of the test data (percent)

# split dataset into sets for testing and training
X = DATA_SET[:, 1:13]  # takes all the 12 features
Y = DATA_SET[:, 0:1]  # takes the label column

# normalised the data through min-max scaling
standard_scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
Scaled_X = standard_scaler.fit_transform(X)

# split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, shuffle=False)
# sklearn ANN
mlp = MLPClassifier(activation='logistic', alpha=1e-05, batch_size=256,
                    beta_1=0.9, beta_2=0.999, early_stopping=False,
                    epsilon=1e-08, hidden_layer_sizes=100,
                    learning_rate='constant', learning_rate_init=0.001,
                    max_iter=1000, momentum=0.9, n_iter_no_change=10,
                    nesterovs_momentum=True, power_t=0.5, random_state=1,
                    shuffle=False, solver='adam', tol=0.0001,
                    validation_fraction=0.1, verbose=False, warm_start=False)
mlp.fit(x_train, y_train.ravel())

print()
print("Starting Training")
print()
print("Accuracy Summary: ")
print('RFT accuracy: (TRAINING)', mlp.score(x_train, y_train))
print('RFT accuracy: (TESTING)', mlp.score(x_test, y_test))


predictions = mlp.predict(x_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print()