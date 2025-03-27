import numpy
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from omelet.misclassification_detection.SPROUTConnector import build_sprout_object, SPROUTGroup
from omelet.utils.classifier_utils import compute_binary_value, compute_multi_value, is_fit


class MisclassificationDetector:
    """
    Class for building objects able to reject predictions according to specific rules
    """

    def __init__(self):
        """
        Constructor
        """
        self.is_fit = False

    def fit(self, proba: numpy.ndarray, y_true: numpy.ndarray, verbose=True):
        """
        Makes the prediction rejection strategy ready to be applied.
        In this case, it identifies ranges in which predictions should be excluded
        :return:
        """
        if proba is None:
            print('Cannot fit with None probas')
        else:
            self.fit_rejector(proba, numpy.argmax(proba, axis=1), y_true, verbose)
            self.is_fit = True

    def fit_rejector(self, proba: numpy.ndarray, y_pred: numpy.ndarray, y_true: numpy.ndarray, verbose=True):
        pass

    def reject_probability(self, test_proba: numpy.ndarray, x_test: numpy.ndarray):
        """
        Findes rejections in a specific test set
        :param x_test: test set
        :param test_proba: the data to apply the strategy to
        :return:
        """
        pass

    def apply(self, test_proba: numpy.ndarray, x_test: numpy.ndarray, test_label: numpy.ndarray = None, reject_tag=None,
              reject_ranges: list = None):
        """
        Applies the prediction rejection strategy to a specific test set
        :param x_test: test set
        :param reject_ranges: ranges to be used for rejecting: if a proba is in range, is rejected
        :param test_label: the predicted labels
        :param reject_tag: the item that corresponds to a prediction rejection
        :param test_proba: the data to apply the strategy to
        :return:
        """
        if test_label is None:
            test_label = numpy.argmax(test_proba, axis=1)
        rejects = self.reject_probability(test_proba, x_test, reject_ranges)
        return numpy.asarray([reject_tag if rejects[i] > 0 else test_label[i] for i in range(len(rejects))])

    def get_name(self) -> str:
        """
        Returns the name of the strategy
        :return:
        """
        return self.__class__.__name__


class ValueAware(MisclassificationDetector):
    """
    Binary value aware strategy to reject predictions
    From Sayin, B., Casati, F., Passerini, A., Yang, J., & Chen, X. (2022). Rethinking and recomputing the value of ML models. arXiv preprint arXiv:2209.15157
    """

    def __init__(self, cost_matrix, reject_cost: int = 0, candidate_thresholds: list = [0.5]):
        """
        Constructor
        :param cost_matrix: the cost matrix (may not be used)
        :param alr: the acceptable level of risk for the strategy
        :param normal_tag: None if multi-class, name of the normal class if binary classification
        """
        MisclassificationDetector.__init__(self)
        self.cost_matrix = cost_matrix
        self.reject_cost = reject_cost
        self.reject_ranges = []
        self.candidate_thresholds = candidate_thresholds
        self.is_binary = None

    def fit_rejector(self, proba: numpy.ndarray, y_pred: numpy.ndarray, y_true: numpy.ndarray, verbose=True):
        """
        Makes the prediction rejection strategy ready to be applied.
        In this case, it identifies ranges in which predictions should be excluded
        :return:
        """
        self.is_binary = len(numpy.unique(y_true)) == 2
        if self.is_binary:
            normal_tag = max(y_true, key=y_true.count)
            # finding the FP thr
            best_fp = 0
            max_value = -numpy.inf
            for t_fp in self.candidate_thresholds:
                # here we define K = fn_c_norm, change it based on task.
                y_rej = self.apply(proba, None, None, None, [t_fp, 0.5])
                value = compute_binary_value(y_rej, y_true, self.cost_matrix, self.reject_cost, None, normal_tag)
                if value > max_value:
                    best_fp = t_fp
                    max_value = value
            # finding the FN thr
            best_fn = 0
            max_value = -numpy.inf
            for t_fn in self.candidate_thresholds:
                # here we define K = fn_c_norm, change it based on task.
                y_rej = self.apply(proba, None, None, None, [0.5, t_fn])
                value = compute_binary_value(y_rej, y_true, self.cost_matrix, self.reject_cost, None,
                                             normal_tag)
                if value > max_value:
                    best_fn = t_fn
                    max_value = value
            self.reject_ranges = [best_fp, best_fn]
        else:
            best_thr = 0
            max_value = -numpy.inf
            for t_misc in self.candidate_thresholds:
                # here we define K = fn_c_norm, change it based on task.
                y_rej = self.apply(proba, None, None, None, [0.5, t_misc])
                value = compute_multi_value(y_rej, y_true, self.cost_matrix, self.reject_cost, None)
                if value > max_value:
                    best_thr = t_misc
                    max_value = value
            self.reject_ranges = [best_thr]

    def reject_probability(self, test_proba: numpy.ndarray, x_test: numpy.ndarray, reject_ranges: list = None):
        """
        Findes rejections in a specific test set
        :param x_test: test set
        :param reject_ranges: ranges to be used for rejecting: if a proba is in range, is rejected
        :param test_proba: the data to apply the strategy to
        :return:
        """
        if reject_ranges is None:
            reject_ranges = self.reject_ranges
        rejects = numpy.full(test_proba.shape[0], 0)
        if self.is_binary:
            # the binary case has two thresholds, one for FP and one for FN
            rejects[test_proba[:, 1] < reject_ranges[0]] = 1
            rejects[test_proba[:, 0] < reject_ranges[1]] = 1
        else:
            # The multi-class case has a single threshold on argmax
            max_proba = numpy.argmax(test_proba, axis=1)
            rejects[max_proba < reject_ranges[0]] = 1
        return rejects


class SufficientlySafe(MisclassificationDetector):
    """
    Binary value aware strategy to reject unsafe predictions
    From Gharib, M., Zoppi, T., & Bondavalli, A. (2022). On the properness of incorporating binary classification machine learning algorithms into safety-critical systems. IEEE Transactions on Emerging Topics in Computing, 10(4), 1671-1686.
    """

    def __init__(self, alr: float = 0.001, max_iterations: int = 5):
        """
        Constructor
        :param alr: the acceptable level of risk for the strategy
        :param normal_tag: None if multi-class, name of the normal class if binary classification
        :param max_iterations: the number of iterations to recursively find thresholds
        """
        MisclassificationDetector.__init__(self)
        self.alr = alr
        self.reject_ranges = []
        self.max_iterations = max_iterations

    def fit_rejector(self, proba: numpy.ndarray, y_pred: numpy.ndarray, y_true: numpy.ndarray, verbose=True):
        """
        Makes the prediction rejection strategy ready to be applied.
        In this case, it identifies ranges in which predictions should be excluded
        :return:
        """
        fns = (y_true != y_pred) * (proba[:, 0] > 0.5)
        fn_probas = sorted(proba[fns, 0])
        i = 0
        while len(fn_probas) > 0:
            residual_fns = self.residual_fns(proba, fns, self.reject_ranges)
            if residual_fns < self.alr:
                # Means that we are able to avoid enough FNs to comply with the ALR
                break
            # Otherwise, we have to modify the reject range (reject more)
            self.reject_ranges = [[0.5, fn_probas.pop(0)]]
            i += 1
        if verbose:
            print("Ended with reject range %s and SSPr=%.3f" % (";".join([str(x) for x in self.reject_ranges]),
                                                                self.sufficiently_safe_value(proba)))

    def residual_fns(self, test_proba: numpy.ndarray, fn_list: numpy.ndarray, reject_ranges: list = None) -> float:
        """
        Returns the % of FNs that are not rejected
        :param test_proba: the probabilities on some test set
        :param fn_list: the list of false negatives
        :param reject_ranges: (optional) custom reject ranges, or self.reject_ranges are used instead
        :return: a percentage (to be compared with ALR)
        """
        fn_count = sum(1 * fn_list)
        rejects = self.reject_probability(test_proba, None, reject_ranges)
        rejected_fn = rejects * fn_list
        return (fn_count - sum(rejected_fn)) / len(fn_list)

    def sufficiently_safe_value(self, test_proba: numpy.ndarray, reject_ranges: list = None) -> float:
        """
        returns the SSPr value of the paper
        :param test_proba: probability of the test set
        :param reject_ranges: (optional) custom reject ranges, or self.reject_ranges are used instead
        :return: SSPr value
        """
        rejects = self.reject_probability(test_proba, None, reject_ranges)
        nssp = sum(rejects)
        ssp = len(rejects) - nssp
        return ssp / (nssp + ssp)

    def reject_probability(self, test_proba: numpy.ndarray, x_test: numpy.ndarray, reject_ranges: list = None):
        """
        Findes rejections in a specific test set
        :param x_test: test set
        :param reject_ranges: ranges to be used for rejecting: if a proba is in range, is rejected
        :param test_proba: the data to apply the strategy to
        :return:
        """
        if reject_ranges is None:
            reject_ranges = self.reject_ranges
        if len(reject_ranges) == 0:
            reject_ranges = [[-1, -1], [2, 2]]
        into_intervals = [numpy.where((my_range[0] <= test_proba[:, 0]) & (test_proba[:, 0] <= my_range[1]), 1, 0) for
                          my_range in reject_ranges]
        into_i = numpy.sum(numpy.asarray(into_intervals), axis=0)
        rejects = 1 * (into_i > 0)
        return rejects

    def get_name(self) -> str:
        """
        Returns the name of the strategy
        :return:
        """
        return self.__class__.__name__ + "(" + str(self.alr) + ")"


class VotingEnsembleRejection(MisclassificationDetector):
    """
    Computes many measures to understand when to reject
    """

    def __init__(self, rejectors: list = []):
        """
        Constructor
        :param rejectors: the list of PredictionRejectors to ensemble
        """
        MisclassificationDetector.__init__(self)
        self.rejectors = rejectors

    def fit_rejector(self, proba: numpy.ndarray, y_pred: numpy.ndarray, y_true: numpy.ndarray, verbose=True):
        """
        Makes the prediction rejection strategy ready to be applied.
        In this case, it identifies ranges in which predictions should be excluded
        :return:
        """
        for rejector in self.rejectors:
            if not rejector.is_fit():
                rejector.fit(proba, y_pred, y_true, verbose)

    def is_fit(self) -> bool:
        """
        :return:
        """
        for rejector in self.rejectors:
            if not rejector.is_fit():
                return False
        return True

    def reject_probability(self, test_proba: numpy.ndarray, x_test: numpy.ndarray, reject_ranges: list = None):
        """
        Findes rejections in a specific test set
        :param x_test: test set
        :param reject_ranges: ranges to be used for rejecting: if a proba is in range, is rejected
        :param test_proba: the data to apply the strategy to
        :return:
        """
        rejs = []
        for rejector in self.rejectors:
            rejs.append(rejector.reject_probability(test_proba, x_test, None))
        rejs = numpy.sum(numpy.vstack(rejs), axis=0)
        r_proba = numpy.average(rejs, axis=1)
        return r_proba

    def get_name(self) -> str:
        """
        Returns the name of the strategy
        :return:
        """
        return self.__class__.__name__ + "(" + "#".join(
            [str(x.get_name()) for x in self.rejectors]) + ")"


class SPROUTRejection(MisclassificationDetector):
    """
    Uses the SPROUT-ML library to reject predictions
    """

    def __init__(self, classifier, x_train, y_train, x_val, y_val, label_names,
                 uncertainty_measures:list = [SPROUTGroup.UM3]):
        """
        Constructor
        """
        MisclassificationDetector.__init__(self)
        self.classifier = classifier
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.uncertainty_measures = uncertainty_measures \
            if uncertainty_measures is not None and len(uncertainty_measures) > 0 else [SPROUTGroup.UM3]
        self.label_names = label_names
        self.sprout = build_sprout_object(x_train, y_train, strategies=self.uncertainty_measures,
                                          label_names=self.label_names)

    def fit_rejector(self, proba: numpy.ndarray, y_pred: numpy.ndarray, y_true: numpy.ndarray, verbose=True):
        """
        Makes the prediction rejection strategy ready to be applied.
        In this case, it identifies ranges in which predictions should be excluded
        :return:
        """
        self.sprout.train_model(self.classifier, self.x_train, self.y_train, self.x_val, self.y_val)

    def is_fit(self) -> bool:
        """
        True if already fit
        :return: boolean
        """
        try:
            if self.classifier is None or self.sprout is None or self.sprout.binary_adjudicator is None:
                return False
            check_is_fitted(self.classifier)
            return True
        except NotFittedError:
            return False

    def reject_probability(self, test_proba: numpy.ndarray, x_test: numpy.ndarray, reject_ranges: list = None):
        """
        Findes rejections in a specific test set
        :param x_test: test set
        :param reject_ranges: ranges to be used for rejecting: if a proba is in range, is rejected
        :param test_proba: the data to apply the strategy to
        :return:
        """
        return self.sprout.predict_misclassifications_probability(x_test, self.classifier, verbose=False)[:, 1]

    def get_name(self) -> str:
        """
        Returns the name of the strategy
        :return:
        """
        return self.__class__.__name__ + "(" + str([x.name for x in self.uncertainty_measures]) + ")"
