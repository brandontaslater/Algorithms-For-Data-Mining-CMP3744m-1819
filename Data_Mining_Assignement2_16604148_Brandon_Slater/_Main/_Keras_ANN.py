# imports:
import os
import time
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import CSVLogger, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix


# removes some of the warning from Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
RUN_MODEL = True

# setting the pd data frame to work within pycharm console interface
desired_width = 500
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 13)


# balances the data set
def balanced_normal_abnormal(FILE_PATH):
    data_set = (np.loadtxt(FILE_PATH, delimiter=",")).transpose()  # read the dataset into a np array, transposes
    normal = (data_set[:, [i for i in range(498)]]).transpose()  # extracts the normal statuses
    abnormal = (data_set[:, [i+498 for i in range(498)]]).transpose()  # extracts the abnormal statuses
    np.random.seed(0), np.random.shuffle(normal)  # shuffles both the classes in the same way
    np.random.seed(0), np.random.shuffle(abnormal)  # creates a new array to store the balanced data
    k_fold = np.zeros((996, 13))
    count = 0  # used for new array
    step = 0  # used for looping the class arrays
    # loops through to the number of samples
    while count < 996:
        k_fold[count] = normal[step]  # the class arrays pass one each time loop into new balances array.
        k_fold[count+1] = abnormal[step]  # the class arrays pass one each time loop into new balances array.
        count += 2  # increments count
        step += 1  # increments step
    return k_fold  # returns the balanced data set


# ====================================================================
# ========================== READ IN DATA ============================
# ====================================================================
# read the dataset into a np array
DATA_SET = balanced_normal_abnormal('Data.csv')

# split dataset into sets for testing and training
X = DATA_SET[:, 1:13]  # takes all the 12 features
Y = DATA_SET[:, 0:1]  # takes the label column

# normalised the data through min-max scaling
standard_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
Scaled_X = standard_scaler.fit_transform(X)

# section selector
section_running = int(input("Please Select a section to run: (1) for single, (2) for k-fold CV, or (3) for K-fold, k-fold:   "))

# ====================================================================
# ======================== HYPERPARAMETERS ===========================
# ====================================================================
CLASSIFICATIONS = 1  # number of reactor classes
LOSS_FUNCTION = 'binary_crossentropy'  # used for categorical classes 2 or more
OPTIMISER = "adam"  # optimiser used with the hidden layer
HIDDEN_LAYER_ACTIVATION = 'sigmoid'  #
OUTPUT_LAYER_ACTIVATION = 'sigmoid'  # logistic function
INPUT_DIMENSION = 12  # number of features input
BATCH_SIZE = 256  # batches processed per epoch
EPOCH = 1000  # the number of epochs (wont reach this because callback break)
TEST_SIZE = 0.10  # the size of the test data (percent)

# runs section 3 in assignment
if section_running == 1:
    # ====================================================================
    # ============================= Section 3 ============================
    # ====================================================================
    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    print("------ Section 3: Training ANN for Normal or Abnormal Data -------")
    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    print()
    print("About to start training")
    print()

    # Wait for 3 seconds
    time.sleep(3)

    NEURONS = 100  # number of neurons in the hidden layer

    # File paths
    MODEL_NAME = "Model"
    TRAINING_LOGS_FILE = MODEL_NAME + ".csv"
    MODEL_SUMMARY_FILE = MODEL_NAME + ".txt"
    MODEL_SAVE_FILE = MODEL_NAME + ".h5"

    # splits the features and labels into train and test data
    # randomises the data with random seed
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, shuffle=False)
    # early epoch stopping criteria if after 50 epochs the min loss hasn't decreased stop model learning
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

    # creating model
    model = Sequential()
    # input data into hidden layer
    model.add(Dense(NEURONS, input_dim=INPUT_DIMENSION, activation=HIDDEN_LAYER_ACTIVATION))
    # input data into hidden layer
    model.add(Dense(NEURONS, activation=HIDDEN_LAYER_ACTIVATION))
    # input data into hidden layer
    model.add(Dense(NEURONS, activation=HIDDEN_LAYER_ACTIVATION))
    # output classification layer using logistic activation function
    model.add(Dense(CLASSIFICATIONS, activation='sigmoid'))
    # compile model
    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMISER, metrics=['accuracy'])
    # fit data to model
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(x_test, y_test),
              callbacks=[CSVLogger(TRAINING_LOGS_FILE, append=False, separator=";")])

    # File locations for all output data
    print()
    print("Locations of Files:")
    print("Model Summary .txt: ", str(MODEL_SUMMARY_FILE))
    print("Model Saved .h5: ", str(MODEL_SAVE_FILE))
    print("Training/ Testing Logs: ", str(TRAINING_LOGS_FILE))

    # evaluate the model for train and test
    TRAIN_scores = model.evaluate(x_train, y_train, verbose=0)
    TEST_scores = model.evaluate(x_test, y_test, verbose=0)

    # takes evaluations and takes the final accuracy for train and test
    TRAIN_score_acc = (model.metrics_names[1], TRAIN_scores[1] * 100)
    TEST_score_acc = (model.metrics_names[1], TEST_scores[1] * 100)

    # Model Evaluations
    print()
    print("Accuracies;")
    print("Train"+str(TRAIN_score_acc))
    print("Test"+str(TEST_score_acc))

    # code from https://towardsdatascience.com/
    test = model.predict(x_test)
    test = (test > 0.5)
    cm = confusion_matrix(y_test, test)
    tn, fp, fn, tp = confusion_matrix(y_test, test).ravel()
    print("confusion matrix: ", cm)
    # Precision
    Precision = tp / (tp + fp)
    print("Precision {:0.2f}".format(Precision))
    # Specificity
    Specificity = tn / (tn + fp)
    print("Specificity {:0.2f}".format(Specificity))
    # Sensitivity
    Recall = tp / (tp + fn)
    print("Sensitivity {:0.2f}".format(Recall))
    # F1 Score
    f1 = (2 * Precision * Recall) / (Precision + Recall)
    print("F1 Score {:0.2f}".format(f1))

    # serialize weights to HDF5
    model.save_weights(MODEL_SAVE_FILE)

# runs section 4 in assignment
elif section_running == 2:
    # ====================================================================
    # ============================ Section 4 =============================
    # =========================== CV NEURONS =============================
    # ====================================================================
    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    print("------------- Section 4: 10 Fold Cross Validation ----------------")
    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    print()
    print("About to start training")
    print()

    # Wait for 5 seconds
    time.sleep(3)

    # stores all the accuracies from train and test for the 10 fold cross validation for each parameter test
    # requires running file again with new value
    train_accuracy_cv = []
    test_accuracy_cv = []

    # k-fold confusion matrix arrays
    normal_ = []
    abnormal_ = []
    CV_no = 0

    # initialises the number of folds
    kf = KFold(n_splits=10)
    # loops through 10 fold cross validations
    for train, test in kf.split(DATA_SET):
        # splits the features and labels into train and test data
        train_data = np.array(DATA_SET)[train]
        test_data = np.array(DATA_SET)[test]
        x_train, y_train = train_data[:, 1:13], train_data[:, 0:1]  # takes all the 12 features, takes the label column
        x_test, y_test = test_data[:, 1:13], test_data[:, 0:1]  # takes all the 12 features, takes the label column

        CV_no += 1  # counts the numbver of CV
        NEURONS = 500  # number of neurons in the hidden layer

        # File paths
        MODEL_NAME = "Model_" + str(CV_no) + "_Neuron" + str(NEURONS)
        TRAINING_LOGS_FILE = MODEL_NAME + ".csv"
        MODEL_SUMMARY_FILE = MODEL_NAME + ".txt"
        MODEL_SAVE_FILE = MODEL_NAME + ".h5"

        # early epoch stopping criteria if after 50 epochs the min loss hasn't decreased stop model learning
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
        # creating model
        model = Sequential()
        # input data into hidden layer
        model.add(Dense(NEURONS, input_dim=INPUT_DIMENSION, kernel_initializer='normal', activation=HIDDEN_LAYER_ACTIVATION))
        # output classification layer using logistic activation function
        model.add(Dense(CLASSIFICATIONS, kernel_initializer='normal', activation=OUTPUT_LAYER_ACTIVATION))
        # compile and fit model
        model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMISER, metrics=['accuracy'])
        # fit data to model
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(x_test, y_test),
                  callbacks=[CSVLogger(TRAINING_LOGS_FILE, append=False, separator=";")])

        # creates a file with the architecture of the ANN
        with open(MODEL_SUMMARY_FILE, "w") as fh:
            model.summary(print_fn=lambda line: fh.write(line + "\n"))

        # evaluate the model for train and test
        TRAIN_scores = model.evaluate(x_train, y_train, verbose=0)
        TEST_scores = model.evaluate(x_test, y_test, verbose=0)

        # takes evaluations and takes the final accuracy for train and test
        TRAIN_score_acc = (model.metrics_names[1], TRAIN_scores[1] * 100)
        TEST_score_acc = (model.metrics_names[1], TEST_scores[1] * 100)

        # serialize weights to HDF5
        model.save_weights(MODEL_SAVE_FILE)

        # stores the accuracies for each cross validation for train and test
        train_accuracy_cv.append(TRAIN_score_acc[1])
        test_accuracy_cv.append(TEST_score_acc[1])

        # predicts from test data
        test = model.predict(x_test)
        test = (test > 0.5)
        # plotting confusion matrix
        cm = confusion_matrix(y_test, test)
        # adds to bug confusion matrix
        normal_.append(list(cm[0]))
        abnormal_.append(list(cm[1]))

    # prints all CV (10) accuracy
    print()
    print("10 Fold Cross Validation Accuracies;")
    print("Training Accuracies:")
    print(train_accuracy_cv)
    print(), print('Avg Train: ', (sum(train_accuracy_cv)/10)), print()
    print("Testing Accuracies:")
    print(test_accuracy_cv)
    print(), print('Avg Test: ', (sum(test_accuracy_cv)/10)), print()
    print("Normal confusion:")
    print('True | False')
    print(normal_)
    print("Abnormal confusion:")
    print('False | True')
    print(abnormal_)

    # calculates confusion matrix for normal and abnormal
    confusion_mat = np.zeros((2, 2))
    # loops through the size of the k-fold
    for i in range(10):
        # adds up all the k fold iterations
        confusion_mat[0, 0] += normal_[i][0]
        confusion_mat[0, 1] += normal_[i][1]
        confusion_mat[1, 0] += abnormal_[i][0]
        confusion_mat[1, 1] += abnormal_[i][1]
    # gets average for each within matrix
    confusion_mat[0, 0] /= 10
    confusion_mat[0, 1] /= 10
    confusion_mat[1, 0] /= 10
    confusion_mat[1, 1] /= 10

    print("K-fold Confusion Matrix:")
    print(confusion_mat)

# runs experimental section
elif section_running == 3:
    # ====================================================================
    # ============================ Section 5 =============================
    # =========================== CV NEURONS =============================
    # ====================================================================
    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    print("-------------------- Section 5: Experimental ---------------------")
    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    print()
    print("About to start training")
    print()
