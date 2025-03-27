import numpy
import sklearn.metrics
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from omelet.classifiers.AbstractClassifier import AbstractClassifier
from omelet.misclassification_detection.MisclassificationDetector import MisclassificationDetector
from omelet.utils.classifier_utils import is_fit, get_classifier_name


class FailControlledClassifier(AbstractClassifier):

    def __init__(self, clf, misc_det:MisclassificationDetector,  X_val=None, y_val=None,
                 alr: float = 0.001, max_thr_iterations:int = 15):
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
        self.rej_thr = None
        self.train_metrics = None

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
        last_acceptable_thr = -1
        iter = 0
        while iter < self.max_thr_iterations:
            aw, ew, phi = self.compute_misc_percentage(clf_preds, rej_probas, tmp_rej_thr, self.y_val)
            if ew < self.alr:
                last_acceptable_thr = tmp_rej_thr
                self.train_metrics = {'aw':aw, 'ew':ew, 'phi':phi}
                lower_bound = tmp_rej_thr
                tmp_rej_thr = (tmp_rej_thr + upper_bound) / 2
            else:
                upper_bound = tmp_rej_thr
                tmp_rej_thr = (tmp_rej_thr + lower_bound) / 2
            iter += 1
        if last_acceptable_thr <= 0 or self.train_metrics['aw'] == 0:
            if verbose:
                print("This FCC cannot meet the ALR")
        else:
            self.rej_thr = last_acceptable_thr

    def is_fcc_meeting_alr(self):
        """
        True if the FCC meets the ALR requirements
        :return:
        """
        return self.rej_thr is not None

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
        acc = sum(preds_with_reject == y_true)/len(y_true)
        omissions = numpy.average(preds_with_reject == None)
        return acc, 1.0-acc-omissions, omissions

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
        return numpy.where(rej_mask == False, preds, None)

    def get_name(self):
        """
        Prints the name of the FCC
        :return:
        """
        return "FCC(" + get_classifier_name(self.clf) + "," + self.misc_det.get_name() + "," + str(self.alr) + ")"
