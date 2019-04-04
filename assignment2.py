
import pandas as pd
import matplotlib.pyplot as plt
from GaussianNBClassifier import GaussianNBClassifier


# load training dataset
filename = 'data_training_set.txt'
train_data = pd.read_csv(filename, sep=',', header=None)

# load test dataset
test_file ='test_set.txt'
test_set = pd.read_csv(test_file, sep=',', header=None)

#find distinct class number and assign the variables
[first_class_label, second_class_label] = sorted(train_data.iloc[:,1].unique())

# filter the dataframe by class label
first_class = train_data[train_data.iloc[:, 1] == first_class_label]
second_class = train_data[train_data.iloc[:, 1] == second_class_label]

# visualize the training dataset
plt.scatter(first_class.iloc[:, 0], first_class.iloc[:, 1], marker='x', label='Class 1')
plt.scatter(second_class.iloc[:, 0], second_class.iloc[:, 1], marker='o', label='Class 2')
plt.title("Training Set")
plt.xlabel("Length of the fish")
plt.ylabel("Fish Classes")
plt.show()

# create an instance of classifier
clf = GaussianNBClassifier(train_data)
# make predictions from test dataset
predictions = clf.predict(test_set.iloc[:, 0])
# calculate accuracy of classifier on test dataset
clf.accuracy(predictions, test_set.iloc[:,1])
# draw confussion matrix of predictions
clf.get_confussion_matrix(predictions, test_set.iloc[:,1])