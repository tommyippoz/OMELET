# Support libs
import copy
import os
import random

# Works only with anomaly detection (no multi-class)
# ------- GLOBAL VARS -----------
import numpy as numpy
import pandas
import sklearn
from confens.classifiers.ConfidenceBagging import ConfidenceBagging
from confens.classifiers.ConfidenceBoosting import ConfidenceBoosting
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier

from omelet.classifiers.FailControlledClassifier import FailControlledClassifier, FCCEnsemble
from omelet.misclassification_detection.MisclassificationDetector import SPROUTGroup, SPROUTRejection
from omelet.utils.classifier_utils import get_classifier_name, compute_omission_metrics
from omelet.utils.dataset_utils import read_binary_tabular_dataset, read_tabular_dataset
from omelet.utils.general_utils import current_ms

OUT_FOLDER = "output_folder"
# Name of the folder in which look for tabular (CSV) datasets
CSV_FOLDER = "input_folder_icse"
# Name of the column that contains the label in the tabular (CSV) dataset
LABEL_NAME = 'multilabel'
# Name of the 'normal' class in datasets. This will be used only for binary classification (anomaly detection)
NORMAL_TAG = 0
# Name of the file in which outputs of the analysis will be saved
SCORES_FILE = "test_single_fcc.csv"
# Percentage of test data wrt train data
TVT_SPLIT = [0.5, 0.2, 0.3]
# True if debug information needs to be shown
VERBOSE = True
# True if you want to force binary classification
FORCE_BINARY = False
# Cost of rejections
REJECT_TAG = -1
# maximum amount of data
ROW_LIMIT = 100000
# metric tag
METRIC_TAG = 'ew'

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
        XGBClassifier(n_estimators=100),
        LinearDiscriminantAnalysis(),
        Pipeline([("norm", MinMaxScaler()), ("gnb", GaussianNB())]),
        RandomForestClassifier(n_estimators=100),
        LogisticRegression(),
        ExtraTreesClassifier(n_estimators=100),
        ConfidenceBagging(n_base=10, clf=ExtraTreeClassifier()),
        ConfidenceBoosting(n_base=20, relative_boost_thr=0.8, clf=ExtraTreeClassifier(),
                           learning_rate=3),
        ConfidenceBoosting(n_base=20, relative_boost_thr=0.8, clf=RandomForestClassifier(n_estimators=5),
                           learning_rate=3),
    ]
    return base_learners


def get_alrs() -> list:
    """
    Returns the ALRs to be used in the analysis
    :return:
    """
    return [0.005]


def detector_needs_classifier(sg):
    """
    The SPROUTGroup to analyze
    """
    return sg in [SPROUTGroup.UM2, SPROUTGroup.UM3, SPROUTGroup.UM8]


def get_misclassification_detectors(classifier, data_dict: dict, detectors_dict: dict) -> list:
    """
    returns the list of prediction rejection strategies to be used in experiments
    :param cost_matrix: the cost matrix to be used (if value-aware)
    :return: a list of objects
    """
    detector_list = []
    print("\nGathering Misclassification Detectors\n")
    for sg in [SPROUTGroup.UM1, SPROUTGroup.UM2, SPROUTGroup.UM3, SPROUTGroup.UM4, SPROUTGroup.UM8]:
        if detector_needs_classifier(sg):
            misc_det = SPROUTRejection(x_train=data_dict["x_train"], y_train=data_dict["y_train"],
                                       x_val=data_dict["x_val"], y_val=data_dict["y_val"],
                                       classifier=classifier, label_names=data_dict["label_names"],
                                       uncertainty_measures=[sg])
        else:
            if sg not in detectors_dict:
                detectors_dict[sg] = SPROUTRejection(x_train=data_dict["x_train"], y_train=data_dict["y_train"],
                                                     x_val=data_dict["x_val"], y_val=data_dict["y_val"],
                                                     classifier=classifier, label_names=data_dict["label_names"],
                                                     uncertainty_measures=[sg])
            misc_det = detectors_dict[sg]
        detector_list.append(misc_det)
    return detector_list, detectors_dict


def save_fcc_file(dataset_name, alr, suitable_fccs, x, y, tag):
    to_print = pandas.DataFrame(x)
    to_print.columns = ["feature_" + str(i+1) for i in range(0, x.shape[1])]
    to_print["true_label"] = y
    for fcc in suitable_fccs:
        fcc_name = fcc.get_name()
        start_time = current_ms()
        y_pred = fcc.predict(x)
        inf_time = current_ms() - start_time
        y_proba = fcc.predict_proba(x)
        to_print[fcc_name + "_pred"] = y_pred
        to_print[fcc_name + "_proba"] = [";".join([str(a) for a in x]) for x in y_proba]
        to_print[fcc_name + "_time"] = numpy.full(len(y_pred), inf_time*1.0/len(y_pred))
    filename = dataset_name + "@ALR=" + str(alr) + "@" + tag + ".csv"
    to_print.to_csv(os.path.join(OUT_FOLDER, filename), index=False)


# ----------------------- MAIN ROUTINE ---------------------
# This script replicates experiments done for testing the robustness of confidence ensembles
if __name__ == '__main__':

    # This is for checkpointing experiments, otherwise it starts every time from scratch
    exp_hist = None
    if os.path.exists(SCORES_FILE):
        exp_hist = pandas.read_csv(SCORES_FILE, usecols=['dataset_tag', 'fcc_name', 'clf_name', 'alr'])
    else:
        with open(SCORES_FILE, 'w') as file_handler:
            file_handler.write(
                "dataset_tag,clf_name,misc_detector,fcc_name,alr,meets_alr,clf_train_time,detector_train_time,"
                "val_clf_acc,val_fcc_aw,val_fcc_ew,val_fcc_phi,val_fcc_ewans,"
                "test_clf_acc,test_fcc_aw,test_fcc_ew,test_fcc_phi,test_fcc_ewans,"
                "best_fcc_name,best_val_clf_acc,best_val_fcc_aw,best_val_fcc_ew,best_val_fcc_phi,"
                "best_test_clf_acc,best_test_fcc_aw,best_test_fcc_ew,best_test_fcc_phi\n")

    # Iterating over datasets
    for dataset_file in os.listdir(CSV_FOLDER):
        # if file is a CSV, it is assumed to be a dataset to be processed
        if dataset_file.endswith(".csv"):
            dataset_name = dataset_file.replace(".csv", "")
            # Read dataset
            if FORCE_BINARY:
                data_dict = read_binary_tabular_dataset(dataset_name=os.path.join(CSV_FOLDER, dataset_file),
                                                        label_name=LABEL_NAME, limit=ROW_LIMIT,
                                                        train_size=TVT_SPLIT[0], val_size=TVT_SPLIT[1],
                                                        shuffle=True, l_encoding=True, normal_tag="normal")
            else:
                data_dict = read_tabular_dataset(dataset_name=os.path.join(CSV_FOLDER, dataset_file),
                                                 label_name=LABEL_NAME, limit=ROW_LIMIT,
                                                 train_size=TVT_SPLIT[0], val_size=TVT_SPLIT[1],
                                                 shuffle=True, l_encoding=True)

            # Loop for training and testing each classifier
            learners = get_classifiers()
            det_dict = {}
            exp_i = 1
            # Loops over ALRs
            for alr in get_alrs():

                # This is the list that will contain FCCs who meet the ALR requirements
                suitable_fccs = []

                print("---------------------------------------------------------------------")
                print("\t\t\t Exercising with ALR = %s" % str(alr))
                print("---------------------------------------------------------------------")

                if not (exp_hist is not None and ((exp_hist['dataset_tag'] == dataset_name) &
                                                  (exp_hist['alr'] == alr) &
                                                  (exp_hist['clf_name'] == 'Ensemble')).any()):

                    # Loops over Base classifiers
                    best_single_fcc = None
                    for base_clf in learners:

                        clf_name = get_classifier_name(base_clf)
                        # Training the algorithm once to get a model
                        start_time = current_ms()
                        base_clf.fit(data_dict["x_train"], data_dict["y_train"])
                        train_time = current_ms() - start_time
                        print("\n Training classifier %s completed in %d ms" % (clf_name, train_time))
                        clf_val_acc = sklearn.metrics.accuracy_score(data_dict["y_val"],
                                                                     base_clf.predict(data_dict["x_val"]))
                        clf_test_acc = sklearn.metrics.accuracy_score(data_dict["y_test"],
                                                                      base_clf.predict(data_dict["x_test"]))
                        print("\t Validation Accuracy: %.5f" % clf_val_acc)
                        print("\t Test Accuracy: %.5f" % clf_test_acc)

                        # Creates FCCs and checks if they meet ALR (ew < ALR).
                        # Keeps track of the one with highest aw and ew < ALR for logging
                        det_list, det_dict = get_misclassification_detectors(base_clf, data_dict, det_dict)
                        for misc_detector in det_list:
                            fcc = FailControlledClassifier(base_clf, misc_detector, data_dict["x_val"],
                                                           data_dict["y_val"],
                                                           alr, 15, REJECT_TAG, METRIC_TAG)
                            fcc.fit(data_dict["x_train"], data_dict["y_train"])
                            val_fcc_metrics = \
                                compute_omission_metrics(data_dict["y_val"], fcc.predict(data_dict["x_val"]),
                                                         reject_tag=REJECT_TAG)
                            test_fcc_metrics = \
                                compute_omission_metrics(data_dict["y_test"], fcc.predict(data_dict["x_test"]),
                                                         reject_tag=REJECT_TAG)
                            if fcc.is_fcc_meeting_alr():
                                suitable_fccs.append(fcc)
                                print("\t %s MEETS the desired ALR=(%.5f<%s) ON THE VALIDATION, aw=%.5f, phi=%.5f, ew=%.5f, ew_ans=%.5f" %
                                      (fcc.get_name(), val_fcc_metrics[METRIC_TAG], str(alr), val_fcc_metrics['aw'],
                                       val_fcc_metrics['phi'], val_fcc_metrics['ew'], val_fcc_metrics['ew_ans']))
                                if best_single_fcc is None or best_single_fcc[1]['aw'] < val_fcc_metrics['aw']:
                                    best_single_fcc = [fcc, val_fcc_metrics, test_fcc_metrics]
                            else:
                                print("\t %s DOES NOT MEET the desired ALR=%s" % (fcc.get_name(), str(alr)))

                            # Checks if FCC was already logged in results
                            if not (exp_hist is not None and (((exp_hist['dataset_tag'] == dataset_name) &
                                                               (exp_hist['fcc_name'] == fcc.get_name())).any())):
                                with open(SCORES_FILE, 'a') as file_handler:
                                    file_handler.write(
                                        dataset_name + "," + clf_name + "," + misc_detector.get_name() + "," +
                                        fcc.get_name() + "," + str(alr) + "," + str(fcc.is_fcc_meeting_alr()) + "," +
                                        str(train_time) + "," + str(misc_detector.train_time) + "," +
                                        str(clf_val_acc) + "," + str(val_fcc_metrics['aw']) + "," +
                                        str(val_fcc_metrics['ew']) + "," + str(val_fcc_metrics['phi']) + "," +
                                        str(val_fcc_metrics['ew_ans']) + "," +
                                        str(clf_test_acc) + "," + str(test_fcc_metrics['aw']) + "," +
                                        str(test_fcc_metrics['ew']) + "," + str(test_fcc_metrics['phi']) + "," +
                                        str(test_fcc_metrics['ew_ans']))
                                    file_handler.write("\n")

                    # Saving Data
                    save_fcc_file(dataset_name, alr, suitable_fccs, data_dict["x_train"], data_dict["y_train"], "TRAIN")
                    save_fcc_file(dataset_name, alr, suitable_fccs, data_dict["x_val"], data_dict["y_val"], "VALIDATION")
                    save_fcc_file(dataset_name, alr, suitable_fccs, data_dict["x_test"], data_dict["y_test"], "TEST")

