import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix


# setting the pd data frame to work within pycharm console interface
desired_width = 320
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


# plots the importance of each feature as a percentage
def plot_feature_influence(RFC, X, x_train): # Random forest Class, X features for full data set and training data set
    # extracts the percentage of importance from each figure
    importance = RFC.feature_importances_
    # calculates standard deviation of each feature
    std = np.std([tree.feature_importances_ for tree in RFC.estimators_], axis=0)
    # gets the indices of the a sorted importance
    indices = np.argsort(importance)[::-1]  #

    print()
    print("Feature ranking: ")
    # Print the feature ranking
    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importance[indices[f]]))

    # Plot the feature importance of the forest
    plt.figure()
    plt.title("Feature importance")
    plt.bar(range(x_train.shape[1]), importance[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(x_train.shape[1]), indices)
    plt.xlim([-1, x_train.shape[1]])
    plt.xlabel('Feature')
    plt.ylabel('Percent')
    plt.show()


# grid search for hypertuning
def random_grid(X, Y):  # features and labels
    # Best Parameters:
    # 'n_estimators': 223, 'min_samples_split': 2, 'min_samples_leaf': 1,
    # 'max_features': 'sqrt','max_depth': None, 'bootstrap': False
    # -----------------
    # splits the features and labels into train and test data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, shuffle=False)
    # empty random forest class
    RFC = RandomForestClassifier()
    # ranges for all hyper testing
    n_estimators = [int(x) for x in np.linspace(start=1, stop=2000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [1, 2, 5, 10]
    min_samples_leaf = [1, 2, 10]
    bootstrap = [True, False]
    # the grid for all the parameters to be tested
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    # repeated until optimal parameters are found
    # start hyper-tuning
    rf_random = RandomizedSearchCV(estimator=RFC, param_distributions=random_grid, n_iter=100,
                                   cv=10, verbose=2, n_jobs=-1)
    # test data against parameter and runs next
    rf_random.fit(x_train, y_train)
    # outputs the best parameters for the data set
    print(rf_random.best_params_)


# ====================================================================
# ========================== READ IN DATA ============================
# ====================================================================
# read the dataset into a np array
DATA_SET = balanced_normal_abnormal('Data.csv')
# split dataset into sets for testing and training
X = DATA_SET[:, 1:13]  # takes all the 12 features
Y = DATA_SET[:, 0:1]  # takes the label column

standard_scaler = MinMaxScaler()
Scaled_X = standard_scaler.fit_transform(X)

# ====================================================================
# ========================== Hyperparameters =========================
# ====================================================================
min_samples_split = 2
max_features = 'sqrt'
max_depth = None
bootstrap = True

# ====================================================================
# ============================= Section 3 ============================
# ====================================================================
no_trees = 100  # tuning parameter for the number of trees
min_samples_leaf = 1  # tuning parameter for the minimum number of leafs per sample (1 or 10)

print()
print("------------------------------------------------------------------")
print("------------------------------------------------------------------")
print("----- Section 3: Training Forest for Normal or Abnormal Data -----")
print("------------------------------------------------------------------")
print("------------------------------------------------------------------")
print()
print("About to start training")

# splits the features and labels into train and test data
# randomises the data with random seed
x_train, x_test, y_train, y_test = train_test_split(Scaled_X, Y, test_size=0.10, shuffle=False)
# setting the RFC class for the all the hyperparameters
RFC = RandomForestClassifier(bootstrap=bootstrap, class_weight=None, criterion='gini',
                             max_depth=max_depth, max_features=max_features, max_leaf_nodes=None,
                             min_impurity_decrease=0.0, min_impurity_split=None,
                             min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                             min_weight_fraction_leaf=0.0, n_estimators=no_trees, n_jobs=None,
                             oob_score=True, random_state=None, verbose=0, warm_start=False)

# fit the data to the model architecture
RFC.fit(x_train, y_train.ravel())

print()
print("Starting Training")
print()
print("Accuracy Summary: ")
print('RFT accuracy: (TRAINING)', RFC.score(x_train, y_train))
print('RFT accuracy: (TESTING)', RFC.score(x_test, y_test))
print('RFT OOB accuracy:', RFC.oob_score_)

# code from https://towardsdatascience.com/
test = RFC.predict(x_test)
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


# ====================================================================
# ============================= Section 4 ============================
# ====================================================================
no_trees = 100  # tuning parameter for the number of trees
min_samples_leaf = 1  # tuning parameter for the minimum number of leafs per sample (1 or 10)
print()
print("------------------------------------------------------------------")
print("------------------------------------------------------------------")
print("------------- Section 4: 10 Fold Cross Validation ----------------")
print("------------------------------------------------------------------")
print("------------------------------------------------------------------")
print()
print("Starting Training")
print()

# stores all the accuracies from train and test for the 10 fold cross validation for each parameter test
# requires running file again with new value
train_accuracy_cv = []
test_accuracy_cv = []
oob_accuracy_cv = []

kf = KFold(n_splits=10)  # number of K folds

# storage for confusion matrix
normal_ = []
abnormal_ = []
CV_no = 0  # counter for cv no

# loops through 10 fold cross validations
for train, test in kf.split(DATA_SET):
    train_data = np.array(DATA_SET)[train]  #
    test_data = np.array(DATA_SET)[test]  #
    x_train, y_train = train_data[:, 1:13], train_data[:, 0:1]  # takes all the 12 features | takes the label column
    x_test, y_test = test_data[:, 1:13], test_data[:, 0:1]  # takes all the 12 features | takes the label column
    CV_no += 1  # increments the cv count
    # setting the RFC class for the all the hyperparameters
    RFC = RandomForestClassifier(bootstrap=bootstrap, class_weight=None, criterion='gini',
                             max_depth=max_depth, max_features=max_features, max_leaf_nodes=None,
                             min_impurity_decrease=0.0, min_impurity_split=None,
                             min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                             min_weight_fraction_leaf=0.0, n_estimators=no_trees, n_jobs=None,
                             oob_score=True, random_state=None, verbose=0, warm_start=False)
    # fit the data to the model architecture
    RFC.fit(x_train, y_train.ravel())
    # stores the accuracies for each cross validation for train and test
    train_accuracy_cv.append(RFC.score(x_train, y_train))
    test_accuracy_cv.append(RFC.score(x_test, y_test))
    oob_accuracy_cv.append(RFC.oob_score_)

    # gets test accuracy
    test = RFC.predict(x_test)
    test = (test > 0.5)
    # creation of confusion matrix
    cm = confusion_matrix(y_test, test)

    # adds to normal and abnoral data storage for each fold
    normal_.append(list(cm[0]))
    abnormal_.append(list(cm[1]))
    print("Cross Validation Split: ", str(CV_no))
    print()

print()
print("Finished Training")
print()
print("10 Fold Cross Validation")
print("Accuracy Summary: ")
print('RFT Training Accuracy: ')
print(train_accuracy_cv)
print(), print('Avg Train: ', (sum(train_accuracy_cv)/10)), print()
print('RFT Testing Accuracy: ')
print(test_accuracy_cv)
print(), print('Avg Test: ', (sum(test_accuracy_cv)/10)), print()
print('OOB Accuracy: ')
print(oob_accuracy_cv)
print(), print('Avg OOB: ', (sum(oob_accuracy_cv)/10)), print()

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



