# Support libs
import os
import random

# Works only with anomaly detection (no multi-class)
# ------- GLOBAL VARS -----------
import numpy as numpy
import pandas
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier

from omelet.classifiers.FailControlledClassifier import FailControlledClassifier
from omelet.misclassification_detection.MisclassificationDetector import SPROUTGroup, SPROUTRejection
from omelet.utils.classifier_utils import get_classifier_name
from omelet.utils.dataset_utils import read_binary_tabular_dataset, read_tabular_dataset
from omelet.utils.general_utils import current_ms

# Name of the folder in which look for tabular (CSV) datasets

CSV_FOLDER = "input_folder"
# Name of the column that contains the label in the tabular (CSV) dataset
LABEL_NAME = 'multilabel'
# Name of the 'normal' class in datasets. This will be used only for binary classification (anomaly detection)
NORMAL_TAG = 0
# Name of the file in which outputs of the analysis will be saved
SCORES_FILE = "test_value.csv"
# Percentage of test data wrt train data
TVT_SPLIT = [0.5, 0.2, 0.3]
# True if debug information needs to be shown
VERBOSE = True
# True if you want to force binary classification
FORCE_BINARY = False
# Cost of rejections
REJECT_COST = 0

# Set random seed for reproducibility
random.seed(42)
numpy.random.seed(42)


# --------- SUPPORT FUNCTIONS ---------------
def get_classifiers() -> list:
    """
    Function to get a learner to use, given its string tag
    :return: the list of classifiers to be trained
    """
    base_learners = [
        ExtraTreeClassifier(),
        # XGBClassifier(n_estimators=100),
        LinearDiscriminantAnalysis(),
        Pipeline([("norm", MinMaxScaler()), ("gnb", GaussianNB())]),
        # RandomForestClassifier(n_estimators=100),
        # LogisticRegression(),
        # ExtraTreesClassifier(n_estimators=100),
    ]
    return base_learners


def get_cost_matrix() -> list:
    """
    Returns the list of cost matrix to be used in the experiments
    :return:
    """
    if FORCE_BINARY:
        return [100, -1, -10000, 1]
    else:
        return [1, -100]


def get_alrs() -> list:
    """
    Returns the ALRs to be used in the analysis
    :return:
    """
    return [0.01, 0.001, 0.0001, 0.00001]


def get_misclassification_detectors(classifier, data_dict: dict) -> list:
    """
    returns the list of prediction rejection strategies to be used in experiments
    :param cost_matrix: the cost matrix to be used (if value-aware)
    :return: a list of objects
    """
    detector_list = []
    for sg in SPROUTGroup:
        detector_list.append(SPROUTRejection(x_train=data_dict["x_train"], y_train=data_dict["y_train"],
                                             x_val=data_dict["x_val"], y_val=data_dict["y_val"],
                                             classifier=classifier, label_names=data_dict["label_names"],
                                             uncertainty_measures=[sg]))
    return detector_list


def predict_fcc_ensemble(fcc_ensemble, x_test, reject_tag=-1):
    """
    Returns the prediction of an ensemble of FCCs
    :param fcc_ensemble: a list of fccs
    :param x_test: a test set
    :param reject_tag: a tag that happens when predictions are rejected
    :return:
    """
    fcc_preds = numpy.zeros(shape=[x_test.shape[0], len(fcc_ensemble)])
    for i in range(0, len(fcc_ensemble)):
        fcc_preds[:, i] = fcc_ensemble[i].predict(x_test)
    ens_pred = numpy.full(fcc_preds.shape[0], reject_tag)
    for i in range(0, fcc_preds.shape[0]):
        for j in range(0, len(fcc_ensemble)):
            if numpy.isfinite(fcc_preds[i, j]):
                ens_pred[i] = fcc_preds[i, j]
                break
    return ens_pred


def compute_omission_metrics(y_true: numpy.ndarray, y_fcc: numpy.ndarray, reject_tag=None) -> dict:
    """
    Assumes that y_fcc may have omissions, labeled as 'reject_tag'
    :param y_true: the ground truth labels
    :param y_fcc: the prediction of the FCC
    :param reject_tag: the tag used to label rejections, default is None
    :return: a dictionary of metrics
    """
    met_dict = {}
    met_dict['phi'] = numpy.count_nonzero(y_fcc == reject_tag) / len(y_true)
    met_dict['alpha_w'] = sum(y_true == y_fcc) / len(y_true)
    met_dict['eps_w'] = 1 - met_dict['phi'] - met_dict['alpha_w']
    return met_dict


# ----------------------- MAIN ROUTINE ---------------------
# This script replicates experiments done for testing the robustness of confidence ensembles
if __name__ == '__main__':

    # This is for checkpointing experiments, otherwise it starts every time from scratch
    exp_hist = None
    if os.path.exists(SCORES_FILE):
        exp_hist = pandas.read_csv(SCORES_FILE, usecols=['dataset_tag', 'clf_name', 'misc_detector', 'alr'])

    # Iterating over datasets
    for dataset_file in os.listdir(CSV_FOLDER):
        # if file is a CSV, it is assumed to be a dataset to be processed
        if dataset_file.endswith(".csv"):
            dataset_name = dataset_file.replace(".csv", "")
            # Read dataset
            if FORCE_BINARY:
                data_dict = read_binary_tabular_dataset(dataset_name=os.path.join(CSV_FOLDER, dataset_file),
                                                        label_name=LABEL_NAME, limit=10000,
                                                        train_size=TVT_SPLIT[0], val_size=TVT_SPLIT[1],
                                                        shuffle=True, l_encoding=True, normal_tag="normal")
            else:
                data_dict = read_tabular_dataset(dataset_name=os.path.join(CSV_FOLDER, dataset_file),
                                                 label_name=LABEL_NAME, limit=10000,
                                                 train_size=TVT_SPLIT[0], val_size=TVT_SPLIT[1],
                                                 shuffle=True, l_encoding=True)

            # Loop for training and testing each classifier
            learners = get_classifiers()
            exp_i = 1

            # Loops over ALRs
            for alr in get_alrs():

                # This is the list that will contain FCCs who meet the ALR requirements
                suitable_fccs = []

                print("---------------------------------------------------------------------")
                print("\t\t\t Exercising with ALR = %s" % str(alr))
                print("---------------------------------------------------------------------")

                # Loops over Base classifiers
                for base_clf in learners:

                    clf_name = get_classifier_name(base_clf)
                    # Training the algorithm once to get a model
                    start_time = current_ms()
                    base_clf.fit(data_dict["x_train"], data_dict["y_train"])
                    train_time = current_ms() - start_time
                    print("\n Training classifier %s completed in %d ms" % (clf_name, train_time))
                    print("\t Validation Accuracy: %.5f" %
                          sklearn.metrics.accuracy_score(data_dict["y_val"], base_clf.predict(data_dict["x_val"])))
                    print("\t Test Accuracy: %.5f" %
                          sklearn.metrics.accuracy_score(data_dict["y_test"], base_clf.predict(data_dict["x_test"])))

                    for misc_detector in get_misclassification_detectors(base_clf, data_dict):

                        fcc = FailControlledClassifier(base_clf, misc_detector, data_dict["x_val"], data_dict["y_val"], alr)
                        fcc.fit(data_dict["x_train"], data_dict["y_train"])
                        if fcc.is_fcc_meeting_alr():
                            suitable_fccs.append(fcc)
                            print("\t %s MEETS the desired ALR=(%.5f<%s) ON THE VALIDATION, aw=%.5f, phi=%.5f" %
                                  (fcc.get_name(), fcc.train_metrics['ew'], str(alr), fcc.train_metrics['aw'], fcc.train_metrics['phi']))
                            fcc.predict(data_dict["x_val"])
                        else:
                            print("\t %s DOES NOT MEET the desired ALR=%s" % (fcc.get_name(), str(alr)))
                            # Loop over calibration strategies (None means no calibration)

                # Here we have a complete list of suitable candidates
                # Evaluation using all available FCCs that meet requirements
                val_fcc_metrics = \
                    compute_omission_metrics(data_dict["y_val"],
                                             predict_fcc_ensemble(suitable_fccs, data_dict["x_val"], reject_tag = -1),
                                             reject_tag=-1)
                test_fcc_metrics = \
                    compute_omission_metrics(data_dict["y_test"],
                                             predict_fcc_ensemble(suitable_fccs, data_dict["x_test"], reject_tag=-1),
                                             reject_tag=-1)
                print("%d FCCs meet the desired ALR\n\t the Ensemble has "
                      "\n\t\t VALIDATION SCORES aw %.5f, ew %.5f, phi %.5f and "
                      "\n\t\t TEST SCORES aw %.5f, ew %.5f, phi %.5f" %
                      (len(suitable_fccs),
                       val_fcc_metrics['alpha_w'], val_fcc_metrics['eps_w'], val_fcc_metrics['phi'],
                       test_fcc_metrics['alpha_w'], test_fcc_metrics['eps_w'], test_fcc_metrics['phi']))

