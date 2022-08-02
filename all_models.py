# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import retrieve_imgs as data
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import SGDClassifier

def train(model, training_set, output_path):
    dataset = data.get_batch(training_set)
    train_X = np.array([dataset[i][0]/255 for i in range(len(dataset))])
    train_Y = np.array([dataset[i][1] for i in range(len(dataset))])
    model.fit(train_X, train_Y, classes=classes)
    with open(output_path, 'wb') as fid:
        pickle.dump(model, fid)
        fid.close()
    return model

def train_out_of_core(model, num_epochs, training_set, output_path, batch_size):
    print(num_epochs)
    for j in range(0, num_epochs):
        print("Epoch number " + str(j))
        random.shuffle(training_set)
        for i in range(0, len(training_set), batch_size):
            print("Running batch " + str(i))
            batch = data.get_batch(training_set[i : i + batch_size])
            train_X = np.array([batch[i][0]/255 for i in range(batch_size)])
            train_Y = np.array([batch[i][1] for i in range(batch_size)])
            model.partial_fit(train_X, train_Y, classes=classes)
    with open(output_path, 'wb') as fid:
        pickle.dump(model, fid)
        fid.close()  
    return model

"""
    Validates 100 examples at a time
"""
def validate(clf, validation_set, output_path):
    y_pred = []
    y_true_label = []
    for i in range(0, len(validation_set), 100):
        print("Validating batch " + str(i))
        batch = data.get_batch(validation_set[i : i + 100])
        X = np.array([batch[j][0]/255 for j in range(0, 100)])
        Y = np.array([batch[j][1] for j in range(0, 100)])
        prediction = clf.predict(X)
        y_pred = np.concatenate((y_pred, prediction), axis=0)
        y_true_label = np.concatenate((y_true_label, Y), axis=0)
    cf_matrix = confusion_matrix(y_true_label, y_pred)
    with open(output_path + '.pkl', 'wb') as fid:
        pickle.dump(cf_matrix, fid)
        fid.close()
    plot_confusion_matrix(cf_matrix, output_path)

def plot_confusion_matrix(cf_matrix, output_path):
     # LTS Labels on the Axes
    axis_labels = [1,2,3,4]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('Predicted LTS')
    ax.set_xlabel('True LTS')
    ax.set_title('Confusion Matrix'); 
    sns.heatmap(cf_matrix, annot=True, cmap='Blues', xticklabels=axis_labels, yticklabels=axis_labels, fmt='d')
    plt.xlabel('Predicted LTS')
    plt.ylabel('True LTS')
    plt.savefig(output_path + 'validate_cnt.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('Predicted LTS')
    ax.set_xlabel('True LTS')
    ax.set_title('Confusion Matrix'); 
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues', xticklabels=axis_labels, yticklabels=axis_labels)
    plt.xlabel('Predicted LTS')
    plt.ylabel('True LTS')
    plt.savefig(output_path + 'validate_pct.png')


names = [
    "Logistic Regression",
    "Nearest Neighbors Uniform Weights",
    "Nearest Neighbors Distance Weights"
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "MLP"
]

classifiers = [
    SGDClassifier(loss='log_loss', random_state=1),
    KNeighborsClassifier(),
    KNeighborsClassifier(weights="distance"), 
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=6),
    MLPClassifier(hidden_layer_sizes = (64, 32), alpha=1e-05, max_iter=1000, solver='sgd')
]


lts_1_splits = [500, 500, 500]
lts_2_splits = [500, 500, 500]
lts_3_splits = [500, 500, 500]
lts_4_splits = [500, 500, 500]
splits = [lts_1_splits, lts_2_splits, lts_3_splits, lts_4_splits]
id_dataset = data.get_ids(splits)
TRAINING_SET_IDS = data.merge_ids(id_dataset, 'train')
VALIDATION_SET_IDS = data.merge_ids(id_dataset, 'val')
TEST_SET_IDS = data.merge_ids(id_dataset, 'test')
classes = np.array([1, 2, 3, 4])
prefix = 'preliminary_models/'
np.save(f'{prefix}training_set.npy', TRAINING_SET_IDS)
np.save(f'{prefix}validation_set.npy', VALIDATION_SET_IDS)
np.save(f'{prefix}test_set.npy', TEST_SET_IDS)

for i in range(len(classifiers)):
    if names[i] == "Logistic Regression":
        model = train_out_of_core(classifiers[i], 1, TRAINING_SET_IDS, prefix + names[i], 100)
    else:
        model = train(classifiers[i], TRAINING_SET_IDS, prefix + names[i])
    validate(model, VALIDATION_SET_IDS, prefix + names[i])
