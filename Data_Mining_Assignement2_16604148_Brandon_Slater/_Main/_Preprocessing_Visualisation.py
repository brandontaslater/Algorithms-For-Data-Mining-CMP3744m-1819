# Dependencies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from pandas.api.types import is_numeric_dtype


# setting the pd data frame to work within pycharm console interface
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 13)


# ====================================================================
# ============================= Section 1 ============================
# ====================================================================
print()
print("------------------------------------------------------------------")
print("------------------------------------------------------------------")
print("---------- Section 1: Pre-processing and Visualisation -----------")
print("------------------------------------------------------------------")
print("------------------------------------------------------------------")
print()
print()

# stores the value to look for with missing data
missing_values = ["n/a", "na", "--", " "]

# import data and check for missing values within list
my_data = pd.read_csv("Assignment2.csv", na_values=missing_values)
# split the data into normal and abnormal
Normal = my_data[my_data.Status == "Normal"]
Abnormal = my_data[my_data.Status == "Abnormal"]

# aesthetic parameters sns
sns.set()

# for counting categorical data
numeric_type = 0
other_type = 0

# iterates over each column and sees whether its numeric or other; counts the total found
for col in my_data.columns:
    # don't check first column as that's categorical data for classifiers
    if col != "Status":
        # if numeric increment by 1
        if is_numeric_dtype(my_data[col]):
            numeric_type += 1
        # if other data type increment by 1
        else:
            other_type += 1
    else:
        other_type += 1


# STATISTICAL SUMMARY
print("Statistic Summaries: ")
print("---------------------")
print()
print("Number of Element in Dataset: "+str(my_data.size))
print("Number of Features: "+str(my_data.shape[1]-1))
print("Number of Samples: "+str(my_data.shape[0]))
print("Number of Continuous Columns in Dataset: ", numeric_type)
print("Number of Categorical Columns in Dataset: ", other_type)
print("Number of Categorical Variables in Dataset: ", len(my_data.Status.unique()))
print("Number of Samples in for Categorical status 'NORMAL': ", len(my_data[my_data.Status == "Normal"]))
print("Number of Samples in for Categorical status 'ABNORMAL': ", len(my_data[my_data.Status == "Abnormal"]))
print("Missing Values in Dataset: {}.".format(my_data.isnull().values.any()))
print()
print(my_data.dtypes)
print()
print(my_data.describe().transpose())  # outputs a table columns with its data type
print()
print(Normal.describe().transpose())  # outputs a table columns with its data type
print()
print(Abnormal.describe().transpose())  # outputs a table columns with its data type
print()

# import the data set into a np array
data_set = (np.loadtxt('Data.csv', delimiter=",")).transpose()
# extract the the 12 features and no the label
whole_set = (data_set[1:13, [i for i in range(996)]]).transpose()
# create the scaler class with a range of 0 and 1
_min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# transform the whole set features to new scale
features_scale = _min_max_scaler.fit_transform(whole_set)

# transpose the data back to column 12 rows 996
data_set = features_scale.transpose()
# split the data into normal and abnormal
normal = (data_set[:, [i for i in range(498)]]).transpose()
abnormal = (data_set[:, [i + 498 for i in range(498)]]).transpose()

# Get Power_range_sensor_1 from both the normal and abnormal
data_power_range_sensor1_original = [Normal.Power_range_sensor_1, Abnormal.Power_range_sensor_1]
data_power_range_sensor1_normalised = [normal[:, 0], abnormal[:, 0]]

# Subplot the data to a box plot for both
plt.figure(1)
plt.boxplot(data_power_range_sensor1_original, labels=["Normal", "Abnormal"])
plt.suptitle('Box Plot: Status vs Power_range_sensor_1')
plt.xlabel('Staus')
plt.ylabel('Power_range_sensor_1')

# Subplot the density plot of the Status and Pressure_sensor_1
plt.figure(3)
pt = sns.kdeplot(Normal.Pressure_sensor_1, shade=True, label="Normal")
pt = sns.kdeplot(Abnormal.Pressure_sensor_1, shade=True, label="Abnormal")
plt.xlim(0, 67.97)
x = [14.99, 14.99]
y = [0, 0.045]
plt.plot(x, y)
plt.suptitle('Density Plot: Status vs Pressure_sensor_1')
plt.xlabel('Pressure_sensor_1')
plt.ylabel('Density')

# Subplot the data to a box plot for both
plt.figure(2)
plt.boxplot(data_power_range_sensor1_normalised, labels=["Normal", "Abnormal"])
plt.suptitle('Normalised Box Plot: Status vs Power_range_sensor_1')
plt.xlabel('Staus')
plt.ylabel('Power_range_sensor_1')

# Subplot the density plot of the Status and Pressure_sensor_1
plt.figure(4)
pt = sns.kdeplot(normal[:, 4], shade=True, label="Normal")
pt = sns.kdeplot(abnormal[:, 4], shade=True, label="Abnormal")
plt.xlim(0, 1)
plt.suptitle('Normalised Density Plot: Status vs Pressure_sensor_1')
plt.xlabel('Pressure_sensor_1')
plt.ylabel('Density')

plt.figure(5)
corr = Normal.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True, annot=True)
plt.tight_layout()

plt.figure(6)
corr = Abnormal.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True, annot=True)
plt.tight_layout()

# Show plots
plt.show()

print()
