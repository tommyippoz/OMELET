from collections.abc import Iterable

import numpy
import pandas
import sklearn.metrics
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from omelet.classifiers.AbstractClassifier import AbstractClassifier
from omelet.misclassification_detection.MisclassificationDetector import MisclassificationDetector
from omelet.utils.classifier_utils import is_fit, get_classifier_name, compute_omission_metrics


class FailControlledClassifier(AbstractClassifier):

    def __init__(self, clf, misc_det: MisclassificationDetector, X_val=None, y_val=None,
                 alr: float = 0.001, max_thr_iterations: int = 15, reject_tag=None):
        """
        Constructor
        """
        AbstractClassifier.__init__(self)
        self.clf = clf if is_classifier(clf) else RandomForestClassifier()
        self.misc_det = misc_det
        self.X_val = X_val
        self.y_val = y_val
        self.alr = alr
        self.max_thr_iterations = max_thr_iterations
        self.rej_thr = -1
        self.train_metrics = None
        self.reject_tag = reject_tag

    def get_reject_thr(self):
        """
        Gets the threshold used to reject
        :return:
        """
        return self.rej_thr

    def fit_classifier(self, X, y, verbose=False):
        """
        Trains the FCC
        :param X: train data
        :param y: train labels
        :return:
        """
        # Trains main classifier if needed
        if not is_fit(self.clf):
            self.clf.fit(X, y)
        clf_probas = self.clf.predict_proba(self.X_val)
        clf_preds = numpy.argmax(clf_probas, axis=1)

        # Trains Misclassification detector if needed
        if not is_fit(self.misc_det):
            self.misc_det.fit(proba=clf_probas, y_true=y, verbose=False)
        rej_probas = self.misc_det.reject_probability(clf_probas, self.X_val)

        # Here we set the rejection threshold to accomplish the desired ALR
        tmp_rej_thr = 1.0
        upper_bound = 1.0
        lower_bound = 0.0
        self.rej_thr = -1
        iter = 0
        while iter < self.max_thr_iterations:
            aw, ew, phi = self.compute_misc_percentage(clf_preds, rej_probas, tmp_rej_thr, self.y_val)
            if ew < self.alr:
                self.rej_thr = tmp_rej_thr
                self.train_metrics = {'aw': aw, 'ew': ew, 'phi': phi}
                lower_bound = tmp_rej_thr
                tmp_rej_thr = (tmp_rej_thr + upper_bound) / 2
            else:
                upper_bound = tmp_rej_thr
                tmp_rej_thr = (tmp_rej_thr + lower_bound) / 2
            iter += 1
        if self.rej_thr <= 0 or self.train_metrics['aw'] == 0:
            if verbose:
                print("This FCC cannot meet the ALR")

    def is_fcc_meeting_alr(self):
        """
        True if the FCC meets the ALR requirements
        :return:
        """
        return self.rej_thr > 0 and self.train_metrics['aw'] > 0

    def reject_probability(self, X):
        """
        returns probability to reject items of test set
        :param X: test set
        :return: array with rejection probability
        """
        probas = self.predict_proba(X)
        return self.misc_det.reject_probability(probas, X)

    def compute_misc_percentage(self, clf_preds, rej_probas, rej_thr, y_true):
        """
        COmputes the residual ew under these set
        :param clf_preds:
        :param rej_probas:
        :param rej_thr:
        :param y_true:
        :return:
        """
        rej_mask = rej_probas > rej_thr
        preds_with_reject = numpy.where(rej_mask == False, clf_preds, None)
        acc = sum(preds_with_reject == y_true) / len(y_true)
        omissions = numpy.average(preds_with_reject == None)
        return acc, 1.0 - acc - omissions, omissions

    def classifier_predict_proba(self, X):
        """
        To be overridden
        :param X: test data
        :return:
        """
        return self.clf.predict_proba(X)

    def predict(self, X):
        """
        Method to compute predict of a classifier.
        Here it needed to be overridden as well
        :return: array of predicted class
        """
        probas = self.predict_proba(X)
        preds = self.classes_[numpy.argmax(probas, axis=1)]
        rej_probas = self.misc_det.reject_probability(probas, X)
        rej_mask = rej_probas > self.rej_thr
        return numpy.where(rej_mask == False, preds, self.reject_tag)

    def get_name(self):
        """
        Prints the name of the FCC
        :return:
        """
        return "FCC(" + get_classifier_name(self.clf) + ";" + self.misc_det.get_name() + ";" + str(self.alr) + ")"


class CSVFailControlledClassifier(FailControlledClassifier):

    def __init__(self, clf_name, misc_det_name, train_df: pandas.DataFrame, test_df: pandas.DataFrame,
                 alr: float = 0.001, max_thr_iterations: int = 15, reject_tag=None):
        """
        Constructor
        """
        FailControlledClassifier.__init__(self, None, None, None, None, alr, max_thr_iterations, reject_tag)
        self.clf_name = clf_name
        self.misc_det_name = misc_det_name
        self.train_df = train_df
        self.test_df = test_df

    def get_reject_thr(self):
        """
        Gets the threshold used to reject
        :return:
        """
        return self.rej_thr

    def fit_classifier(self, X, y, verbose=False):
        """
        Trains the FCC
        :param X: train data
        :param y: train labels
        :return:
        """
        clf_probas = numpy.asarray([numpy.asarray(x.replace("[", "").replace("]", "").split(";"), dtype=float)
                                    for x in self.train_df["probabilities"]])
        clf_preds = self.train_df["predicted_label"].to_numpy()

        # Trains Misclassification detector if needed
        rej_probas = self.csv_reject_probability(self.train_df)

        # Here we set the rejection threshold to accomplish the desired ALR
        tmp_rej_thr = 1.0
        upper_bound = 1.0
        lower_bound = 0.0
        self.rej_thr = -1
        iter = 0
        while iter < self.max_thr_iterations:
            aw, ew, phi = self.compute_misc_percentage(clf_preds, rej_probas, tmp_rej_thr, self.train_df["true_label"])
            if ew < self.alr:
                self.rej_thr = tmp_rej_thr
                self.train_metrics = {'aw': aw, 'ew': ew, 'phi': phi}
                lower_bound = tmp_rej_thr
                tmp_rej_thr = (tmp_rej_thr + upper_bound) / 2
            else:
                upper_bound = tmp_rej_thr
                tmp_rej_thr = (tmp_rej_thr + lower_bound) / 2
            iter += 1
        if self.rej_thr <= 0 or self.train_metrics['aw'] == 0:
            if verbose:
                print("This FCC cannot meet the ALR")

    def is_fcc_meeting_alr(self):
        """
        True if the FCC meets the ALR requirements
        :return:
        """
        return self.rej_thr > 0 and self.train_metrics['aw'] > 0

    def csv_reject_probability(self, df):
        """
        Probabilities of rejection from a specific column
        :param df: the dataframe with data
        :return:
        """
        values = df[self.misc_det_name].to_numpy()
        values -= numpy.min(values)
        values /= numpy.max(values, axis=0)
        return values

    def predict(self, X):
        """
        Method to compute predict of a classifier.
        Here it needed to be overridden as well
        :return: array of predicted class
        """
        preds = X["predicted_label"].to_numpy()
        rej_probas = self.csv_reject_probability(X)
        rej_mask = rej_probas > self.rej_thr
        return numpy.where(rej_mask == False, preds, self.reject_tag)

    def get_name(self):
        """
        Prints the name of the FCC
        :return:
        """
        return "FCC(" + self.clf_name + ";" + self.misc_det_name + ";" + str(self.alr) + ")"


class FCCEnsemble(FailControlledClassifier):

    def __init__(self, fcc_list: Iterable, X_val=None, y_val=None, alr: float = 0.001, reject_tag=None):
        """
        Constructor
        """
        FailControlledClassifier.__init__(self,
                                          fcc_list[0] if fcc_list is not None and len(fcc_list) > 0 else None,
                                          None, X_val, y_val, alr, 1, reject_tag)
        self.fcc_list = fcc_list if fcc_list is not None else []
        self.estimators_ = None

    def fit_classifier(self, X, y, verbose=False):
        """
        Trains the FCC
        :param X: train data
        :param y: train labels
        :return:
        """
        self.estimators_ = []
        # Trains fccs if needed
        for fcc in self.fcc_list:
            if not is_fit(fcc):
                fcc.fit(X, y)
            if fcc.is_fcc_meeting_alr():
                self.estimators_.append(fcc)
        self.estimators_.sort(key=lambda x: x.train_metrics['aw'], reverse=True)

        clf_preds = self.predict(self.X_val)
        aw = sum(clf_preds == self.y_val) / len(self.y_val)
        phi = numpy.average(clf_preds == self.reject_tag)
        ew = 1 - aw - phi
        self.train_metrics = {'aw': aw, 'ew': ew, 'phi': phi}
        if ew < self.alr and aw > 0:
            self.rej_thr = self.alr
        else:
            self.rej_thr = None
            if verbose:
                print("This FCC Ensemble cannot meet the ALR")

    def is_fcc_meeting_alr(self):
        """
        True if the FCC meets the ALR requirements
        :return:
        """
        return self.estimators_ is not None and len(self.estimators_) > 0

    def reject_probability(self, X):
        """
        returns probability to reject items of test set
        :param X: test set
        :return: array with rejection probability
        """
        r_probs = numpy.zeros(X.shape[0])
        if self.estimators_ is not None:
            for fcc in self.estimators_:
                r_probs += fcc.reject_probability(X)
            r_probs /= len(self.estimators_)
        return r_probs

    def classifier_predict_proba(self, X):
        """
        To be overridden
        :param X: test data
        :return:
        """
        fcc_preds = numpy.zeros(shape=[X.shape[0], len(self.estimators_)])
        fcc_probas = numpy.zeros(shape=[X.shape[0], len(self.estimators_), len(self.classes_)])
        for i in range(0, len(self.estimators_)):
            fcc_preds[:, i] = self.estimators_[i].predict(X)
            fcc_probas[:, i, :] = self.estimators_[i].predict_proba(X)
        ens_probas = numpy.full(fcc_preds.shape[0], self.reject_tag)
        for i in range(0, fcc_preds.shape[0]):
            for j in range(0, len(self.estimators_)):
                if fcc_preds[i, j] != self.reject_tag:
                    ens_probas[i] = fcc_probas[i, j, :]
                    break
        return ens_probas

    def predict(self, X):
        """
        Method to compute predict of a classifier.
        Here it needed to be overridden as well
        :return: array of predicted class
        """
        fcc_preds = numpy.full([X.shape[0], len(self.estimators_)], -1, dtype=object)
        for i in range(0, len(self.estimators_)):
            fcc_preds[:, i] = self.estimators_[i].predict(X)
        ens_pred = numpy.full(fcc_preds.shape[0], self.reject_tag, dtype=object)
        for i in range(0, fcc_preds.shape[0]):
            for j in range(0, len(self.estimators_)):
                if fcc_preds[i, j] != self.reject_tag:
                    ens_pred[i] = fcc_preds[i, j]
                    break
        return ens_pred

    def predict_csv(self, tag="train"):
        """
        Method to compute predict of a classifier.
        Here it needed to be overridden as well
        :return: array of predicted class
        """
        if tag == "train":
            fcc_preds = numpy.full([self.estimators_[0].train_df.shape[0], len(self.estimators_)],
                                   self.reject_tag, dtype=object)
        else:
            fcc_preds = numpy.full([self.estimators_[0].test_df.shape[0], len(self.estimators_)],
                                   self.reject_tag, dtype=object)
        for i in range(0, len(self.estimators_)):
            if tag == "train":
                fcc_preds[:, i] = self.estimators_[i].predict(self.estimators_[i].train_df)
            else:
                fcc_preds[:, i] = self.estimators_[i].predict(self.estimators_[i].test_df)
        ens_pred = numpy.full(fcc_preds.shape[0], self.reject_tag, dtype=object)
        for i in range(0, fcc_preds.shape[0]):
            for j in range(0, len(self.estimators_)):
                if fcc_preds[i, j] != self.reject_tag:
                    ens_pred[i] = fcc_preds[i, j]
                    break
        return ens_pred

    def get_name(self):
        """
        Prints the name of the FCC
        :return:
        """
        return "FCCEnsemble(" + str(len(self.estimators_)) + ";" + str(self.alr) + ")"
