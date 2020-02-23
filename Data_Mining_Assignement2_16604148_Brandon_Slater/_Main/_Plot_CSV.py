import csv
import matplotlib.pyplot as plt


# storage for all files for plotting
epoch = []
train_accuracy = []
train_loss = []
validation_accuracy = []
validation_loss = []

# loops through all 10 files
for x in range(10):
    neurons = 500  # number of neurons testes for cv
    MODEL_NAME = "Model_" + str(x + 1) + "_Neuron" + str(neurons)  # file name for model
    # File paths
    TRAINING_LOGS_FILE = MODEL_NAME + ".csv"
    # open file and extract data
    with open(TRAINING_LOGS_FILE) as f:
        reader = csv.reader(f)
        next(reader) # skip header
        data = [r for r in reader]

    # storage for single file
    epoch_single = []
    train_accuracy_single = []
    train_loss_single = []
    validation_accuracy_single = []
    validation_loss_single = []

    # loops through the data from a single file
    for d in data:
        # removes redundant characters
        d = str(d).strip("[']'=")
        # replace those values from string
        newstr = d.replace("[']", "")

        # stores a single input
        a_split = str(d).split(';')
        epoch_single.append(int(a_split[0]))
        train_accuracy_single.append(float(a_split[1]))
        train_loss_single.append(float(a_split[2]))
        validation_accuracy_single.append(float(a_split[3]))
        validation_loss_single.append(float(a_split[4]))

    # stores all the data for plotting
    epoch.append(epoch_single)
    train_accuracy.append(train_accuracy_single)
    train_loss.append(train_loss_single)
    validation_accuracy.append(validation_accuracy_single)
    validation_loss.append(validation_loss_single)

# colours used for ploting
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

# plots training
f1 = plt.figure(1)
for i in range(10):
    # plotting the line 1 points
    plt.plot(epoch[i], train_accuracy[i], label="Train_Accuracy", color=colors[i])
    # plotting the line 2 points
    plt.plot(epoch[i], train_loss[i], label="Train_Loss", color=colors[i])
    # naming the x axis
    plt.xlabel('Number of Epochs')
    # naming the y axis
    plt.ylabel('Percentage')
    # giving a title to my graph
    plt.title('Epoch Training Accuracy & Loss')

# show a legend on the plot
plt.legend()

# plots the validation
f2 = plt.figure(2)
for i in range(10):
    # plotting the line 1 points
    plt.plot(epoch[i], validation_accuracy[i], label="Validation_Accuracy", color=colors[i])
    # plotting the line 2 points
    plt.plot(epoch[i], validation_loss[i], label="Validation_Loss", color=colors[i])
    # naming the x axis
    plt.xlabel('Number of Epochs')
    # naming the y axis
    plt.ylabel('Percentage')
    # giving a title to my graph
    plt.title('Epoch Validation Accuracy & Loss')

# show a legend on the plot
plt.legend()
# function to show the plot
plt.show()

