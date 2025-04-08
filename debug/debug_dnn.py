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

from omelet.classifiers.FailControlledClassifier import FailControlledClassifier, FCCEnsemble, \
    CSVFailControlledClassifier
from omelet.misclassification_detection.MisclassificationDetector import SPROUTGroup, SPROUTRejection
from omelet.utils.classifier_utils import get_classifier_name, compute_omission_metrics
from omelet.utils.dataset_utils import read_binary_tabular_dataset, read_tabular_dataset
from omelet.utils.general_utils import current_ms

# Name of the folder in which look for DNN scores
SCORES_CSV_FOLDER = "dnn_scores"
# Name of the file in which outputs of the analysis will be saved
SCORES_FILE = "dnn_test_value.csv"
# True if debug information needs to be shown
VERBOSE = True
# Cost of rejections
REJECT_TAG = 'REJECT'
# percentage of test/val
VAL_TEST_PERC = 0.4

U_MEASURES = ['MaxProb Calculator',
              'Entropy Calculator',
              'AutoEncoder Loss (conv)',
              'Combined Calculator (ImageClassifier)',
              'Multiple Combined Calculator (7 - IrIrIrIrIrIrIr classifiers)']

# Set random seed for reproducibility
random.seed(42)
numpy.random.seed(42)


# --------- SUPPORT FUNCTIONS ---------------
def get_alrs() -> list:
    """
    Returns the ALRs to be used in the analysis
    :return:
    """
    return [0.01, 0.001, 0.0001]


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


# ----------------------- MAIN ROUTINE ---------------------
# This script replicates experiments done for testing the robustness of confidence ensembles
if __name__ == '__main__':

    # This is for checkpointing experiments, otherwise it starts every time from scratch
    exp_hist = None
    if os.path.exists(SCORES_FILE):
        exp_hist = pandas.read_csv(SCORES_FILE, usecols=['dataset_tag', 'fcc_name'])
    else:
        with open(SCORES_FILE, 'w') as file_handler:
            file_handler.write("dataset_tag,clf_name,misc_detector,fcc_name,alr,meets_alr,"
                               "val_clf_acc,val_fcc_aw,val_fcc_ew,val_fcc_phi,"
                               "test_clf_acc,test_fcc_aw,test_fcc_ew,test_fcc_phi,"
                               "best_fcc_name,best_val_clf_acc,best_val_fcc_aw,best_val_fcc_ew,best_val_fcc_phi,"
                               "best_test_clf_acc,best_test_fcc_aw,best_test_fcc_ew,best_test_fcc_phi\n")

    # Iterating over datasets
    for dataset_name in os.listdir(SCORES_CSV_FOLDER):
        # if file is a CSV, it is assumed to be a dataset to be processed
        if os.path.isdir(os.path.join(SCORES_CSV_FOLDER, dataset_name)):

            # Loops over ALRs
            for alr in get_alrs():

                # This is the list that will contain FCCs who meet the ALR requirements
                suitable_fccs = []

                print("---------------------------------------------------------------------")
                print("\t\t\t %s Exercising with ALR = %s" % (dataset_name, str(alr)))
                print("---------------------------------------------------------------------")

                best_single_fcc = None

                # Iterating over classifiers
                for dnn_csv in os.listdir(os.path.join(SCORES_CSV_FOLDER, dataset_name)):

                    # if file is a CSV, it is assumed to be a dataset to be processed
                    if dnn_csv.endswith(".csv"):
                        clf_name = dnn_csv.replace("CUSTOM_", "").replace(".csv", "")

                        # Fetching data
                        clf_df = pandas.read_csv(os.path.join(SCORES_CSV_FOLDER, dataset_name, dnn_csv))
                        #clf_df = clf_df.sample(frac=1.0)

                        # Adjusting Uncertainty scores
                        for misc_det_name in U_MEASURES:
                            values = clf_df[misc_det_name].to_numpy()
                            values -= numpy.min(values)
                            values /= numpy.max(values, axis=0)
                            if misc_det_name != 'AutoEncoder Loss (conv)':
                                values = 1 - values
                            clf_df[misc_det_name] = values

                        # Splitting
                        clf_df_val = clf_df.iloc[0:int(VAL_TEST_PERC * len(clf_df)), :]
                        clf_df_test = clf_df.iloc[int(VAL_TEST_PERC * len(clf_df)):, :]

                        clf_val_acc = sklearn.metrics.accuracy_score(clf_df_val["true_label"],
                                                                     clf_df_val["predicted_label"])
                        clf_test_acc = sklearn.metrics.accuracy_score(clf_df_test["true_label"],
                                                                      clf_df_test["predicted_label"])
                        print("\n%s\n\t Validation Accuracy: %.5f" % (clf_name, clf_val_acc))
                        print("\t Test Accuracy: %.5f" % clf_test_acc)

                        # Creates FCCs and checks if they meet ALR (ew < ALR).
                        # Keeps track of the one with highest aw and ew < ALR for logging
                        for misc_det_name in U_MEASURES:

                            fcc = CSVFailControlledClassifier(clf_name, misc_det_name, clf_df_val, clf_df_test,
                                                              alr, 15, REJECT_TAG)
                            fcc.fit(clf_df_val[[misc_det_name, misc_det_name]], clf_df_val["true_label"])
                            val_fcc_metrics = \
                                compute_omission_metrics(clf_df_val["true_label"], fcc.predict(clf_df_val),
                                                         reject_tag=REJECT_TAG)
                            test_fcc_metrics = \
                                compute_omission_metrics(clf_df_test["true_label"], fcc.predict(clf_df_test),
                                                         reject_tag=REJECT_TAG)
                            if fcc.is_fcc_meeting_alr():
                                suitable_fccs.append(fcc)
                                print("\t %s MEETS the desired ALR=(%.5f<%s) ON THE VALIDATION, aw=%.5f, phi=%.5f" %
                                      (fcc.get_name(), val_fcc_metrics['ew'], str(alr), val_fcc_metrics['aw'],
                                       val_fcc_metrics['phi']))
                                if best_single_fcc is None or best_single_fcc[1]['aw'] < val_fcc_metrics['aw']:
                                    best_single_fcc = [fcc, val_fcc_metrics, test_fcc_metrics]
                            else:
                                print("\t %s DOES NOT MEET the desired ALR=%s" % (fcc.get_name(), str(alr)))

                            # Checks if FCC was already logged in results
                            if not (exp_hist is not None and (((exp_hist['dataset_tag'] == dataset_name) &
                                                               (exp_hist['fcc_name'] == fcc.get_name())).any())):
                                with open(SCORES_FILE, 'a') as file_handler:
                                    file_handler.write(
                                        dataset_name + "," + clf_name + "," + misc_det_name + "," +
                                        fcc.get_name() + "," + str(alr) + "," + str(fcc.is_fcc_meeting_alr()) + "," +
                                        str(clf_val_acc) + "," + str(val_fcc_metrics['aw']) + "," +
                                        str(val_fcc_metrics['ew']) + "," + str(val_fcc_metrics['phi']) + "," +
                                        str(clf_test_acc) + "," + str(test_fcc_metrics['aw']) + "," +
                                        str(test_fcc_metrics['ew']) + "," + str(test_fcc_metrics['phi']))
                                    file_handler.write("\n")

                # Here we have a complete list of suitable candidates
                # Evaluation using all available FCCs that meet requirements
                ens_fcc = FCCEnsemble(suitable_fccs, clf_df_val, clf_df_val["true_label"], alr, REJECT_TAG)
                ens_fcc.fit(clf_df_val[[misc_det_name, misc_det_name]], clf_df_val["true_label"])
                ens_val_fcc_metrics = \
                    compute_omission_metrics(clf_df_val["true_label"], ens_fcc.predict_csv("train"),
                                             reject_tag=REJECT_TAG)
                ens_test_fcc_metrics = \
                    compute_omission_metrics(clf_df_test["true_label"],
                                             ens_fcc.predict_csv("test"),
                                             reject_tag=REJECT_TAG)
                print("%d FCCs meet the desired ALR\n\t the Ensemble has "
                      "\n\t\t VALIDATION SCORES aw %.5f, ew %.5f, phi %.5f and "
                      "\n\t\t TEST SCORES aw %.5f, ew %.5f, phi %.5f" %
                      (len(suitable_fccs),
                       ens_val_fcc_metrics['aw'], ens_val_fcc_metrics['ew'], ens_val_fcc_metrics['phi'],
                       ens_test_fcc_metrics['aw'], ens_test_fcc_metrics['ew'], ens_test_fcc_metrics['phi']))

                if best_single_fcc is not None:
                    best_name = best_single_fcc[0].get_name()
                    val_fcc_metrics = best_single_fcc[1]
                    test_fcc_metrics = best_single_fcc[2]
                    print("\t whereas the best individual FCC that meets ALR has "
                          "\n\t\t VALIDATION SCORES aw %.5f, ew %.5f, phi %.5f and "
                          "\n\t\t TEST SCORES aw %.5f, ew %.5f, phi %.5f" %
                          (val_fcc_metrics['aw'], val_fcc_metrics['ew'], val_fcc_metrics['phi'],
                           test_fcc_metrics['aw'], test_fcc_metrics['ew'], test_fcc_metrics['phi']))
                else:
                    best_name = "nobody"
                    val_fcc_metrics = {'aw': 0, 'ew': 0, 'phi': 1}
                    test_fcc_metrics = {'aw': 0, 'ew': 0, 'phi': 1}

                if not (exp_hist is not None and (((exp_hist['dataset_tag'] == dataset_name) &
                                                   (exp_hist['fcc_name'] == ens_fcc.get_name())).any())):
                    with open(SCORES_FILE, 'a') as file_handler:
                        file_handler.write(
                            dataset_name + ",Ensemble,None," + ens_fcc.get_name() + "," + str(alr) + "," +
                            str(ens_fcc.is_fcc_meeting_alr()) + "," +
                            str(ens_val_fcc_metrics['aw']) + "," + str(ens_val_fcc_metrics['aw']) + "," +
                            str(ens_val_fcc_metrics['ew']) + "," + str(ens_val_fcc_metrics['phi']) + "," +
                            str(ens_test_fcc_metrics['aw']) + "," + str(ens_test_fcc_metrics['aw']) + "," +
                            str(ens_test_fcc_metrics['ew']) + "," + str(ens_test_fcc_metrics['phi']) + "," +
                            best_name + "," + str(clf_val_acc) + "," + str(val_fcc_metrics['aw']) + "," +
                            str(val_fcc_metrics['ew']) + "," + str(val_fcc_metrics['phi']) + "," +
                            str(clf_test_acc) + "," + str(test_fcc_metrics['aw']) + "," +
                            str(test_fcc_metrics['ew']) + "," + str(test_fcc_metrics['phi']))
                        file_handler.write("\n")
