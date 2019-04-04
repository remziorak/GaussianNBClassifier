
import numpy as np
from prettytable import PrettyTable

class GaussianNBClassifier:

    def __init__(self, train_set):
        self.train_set = train_set
        self.summaries = self.summarize_data(self.train_set)

    def summarize_data(self, train_set):
        """Separate data class and calculate statistics"""
        # find distinct class number and assign the variables
        [first_class_label, second_class_label] = sorted(train_set.iloc[:, 1].unique())

        # filter the dataframe by class label
        first_class = train_set[train_set.iloc[:, 1] == first_class_label]
        second_class = train_set[train_set.iloc[:, 1] == second_class_label]

        first_class_mean = first_class.iloc[:,0].mean()
        second_class_mean = second_class.iloc[:,0].mean()
        first_class_std = first_class.iloc[:, 0].std()
        second_class_std = second_class.iloc[:, 0].std()

        summaries = [[first_class_label, first_class_mean, first_class_std],
                     [second_class_label, second_class_mean, second_class_std]]

        return summaries

    def calculate_probability (self, x, mean, stdev):
        """Normal Probability Density Function"""
        exponent = np.exp(-(np.power(x-mean, 2)/(2*np.power(stdev, 2))))
        return (1 / (np.sqrt(2*np.pi) * stdev)) * exponent

    def calculate_class_probability(self, input_data):
        """Calculate normal PDF for each class"""
        probabilities = {}
        for j in range(len(self.summaries)):
            class_label = self.summaries[j][0]
            mean = self.summaries[j][1]
            std = self.summaries[j][2]
            x = input_data
            probabilities[class_label] = self.calculate_probability(x, mean, std)
        return probabilities

    def predict(self, input_data):
        predictions = []
        for input_variable in input_data:
            a = self.calculate_class_probability(input_variable)
            if a[self.summaries[0][0]] > a[self.summaries[1][0]]:
                predictions.append(self.summaries[0][0])
            else:
                predictions.append(self.summaries[1][0])

        return predictions

    def accuracy(self, y_test, y_predicted):
        """Compute the accuracy of a classification"""
        correct_prediction = 0
        wrong_prediction = 0
        total_prediction = 0
        for i in range(len(y_test)):
            if y_test[i] == y_predicted[i]:
                correct_prediction += 1
            else:
                wrong_prediction += 1
            total_prediction += 1
        accuracy = (correct_prediction / float(total_prediction)) * 100.0
        print("Accuracy :",accuracy)

    def get_confussion_matrix(self,y_test, y_predicted):
        """Compute confusion matrix to evaluate the accuracy of a classification"""
        tp, fn, fp, tn = 0, 0, 0, 0
        for i in range(len(y_test)):
            if y_predicted[i] == self.summaries[0][0]:
                if y_test[i] == self.summaries[0][0]:
                    tp += 1
                else:
                    fp += 1
            else:
                if y_test[i] == self.summaries[0][0]:
                    fn += 1
                else:
                    tn += 1
        self.draw_confusion_matrix(tp, fn, fp, tn)


    def draw_confusion_matrix(self, tp, fn, fp, tn):
        """
        Draw a confusion matrix as following.

                            Confusion Matrix
        +----------------+---------------------+---------------------+
        |                | Predicted Class = 1 | Predicted Class = 2 |
        +----------------+---------------------+---------------------+
        | True Class = 1 |   True Positive     |   False Positive    |
        | True Class = 2 |   False Negative    |   True Negative     |
        +----------------+---------------------+---------------------+

        """
        confusion_table = PrettyTable()
        confusion_table.field_names = ["", "Predicted Class = 1", "Predicted Class = 2"]
        confusion_table.add_row(["True Class = 1", tp, fp])
        confusion_table.add_row(["True Class = 2", fn, tn])
        print("\n\t\t\t\t\t Confusion Matrix")
        print(confusion_table)

