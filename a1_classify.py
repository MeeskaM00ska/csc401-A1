#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

# set the random state for reproducibility
import numpy as np

np.random.seed(401)

classifier_list = [
    SGDClassifier(),
    GaussianNB(),
    RandomForestClassifier(max_depth=5, n_estimators=10),
    MLPClassifier(),
    AdaBoostClassifier()
]
classifier_dict = {
    0: "SGDClassifier", 1: "GaussianNB", 2: "RandomForestClassifier",
    3: "MLPClassifier", 4: "AdaBoostClassifier"
}


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    sum_c = C.sum()
    if sum_c == 0:
        return 0
    else:
        return C.trace() / sum_c


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    return np.diag(C) / C.sum(axis=1)


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    return np.diag(C) / np.sum(C, axis=0)


def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:
       i: int, the index of the supposed best classifier
    '''
    iBest = 0
    c_acc = 0.0
    method_index = -1
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        for c in classifier_list:
            print(f'model {c} begins')
            method_index += 1
            classifier_name = classifier_dict[method_index]

            c.fit(X_train, y_train)
            pred = c.predict(X_test)
            conf_matrix = confusion_matrix(y_test, pred)
            acc = accuracy(conf_matrix)
            recal = recall(conf_matrix)
            preci = precision(conf_matrix)

            if not acc < c_acc:
                iBest = classifier_list.index(c)
                c_acc = acc
                # For each classifier, compute results and write the following output:
            outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in recal]}\n')
            outf.write(
                f'\tPrecision: {[round(item, 4) for item in preci]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')
    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''

    best_method = None
    data_amount = [1000, 5000, 10000, 15000, 20000]
    if iBest == 0:
        best_method = classifier_list[0]
    elif iBest == 1:
        best_method = classifier_list[1]
    elif iBest == 2:
        best_method = classifier_list[2]
    elif iBest == 3:
        best_method = classifier_list[3]
    elif iBest == 4:
        best_method = classifier_list[4]
    acc = []
    for d in data_amount:
        best_method.fit(X_train[:d], y_train[:d])
        pred = best_method.predict(X_test)
        conf_matrix = confusion_matrix(y_test, pred)
        single_acc = accuracy(conf_matrix)
        acc.append(single_acc)

    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        for i in range(0, 5):
            outf.write(f'{data_amount[i]}: {acc[i]:.4f}\n')
        outf.write("By the output, the larger number of training data, the more accuracy it will be.Increasing the number of training samples is significant to the growth, but when the number reaches convergence, the change brought by increasing the number of samples is not obvious.")
    y_1k = y_train[:1000]
    X_1k = X_train[:1000]
    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('TODO Section 3.3')

    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.

        # for each number of features k_feat, write the p-values for
        # that number of features:
        # outf.write(f'{k_feat} p-values: {[format(pval) for pval in p_values]}\n')

        # outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        # outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        # outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        # outf.write(f'Top-5 at higher: {top_5}\n')
        pass


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
        '''
    print('TODO Section 3.4')

    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        # for each fold:
        #     outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        # outf.write(f'p-values: {[format(pval) for pval in p_values]}\n')
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2",
                        required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    # TODO: load data and split into train and test.
    data = (np.load(args.input))[(np.load(args.input)).files[0]]
    x = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=401)

    # TODO : complete each classification experiment, in sequence.
    print(f'class31 begins')
    iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)
    print(f'class32 begins')
    X_1k, y_1k = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)
