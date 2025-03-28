import numpy

from omelet.metrics.DiversityMetric import QStatDiversity, SigmaDiversity, Disagreement


def get_default():
    return [QStatMetric(), SigmaMetric(), CoupleDisagreementMetric(),
            DisagreementMetric(), SharedFaultMetric()]


class EnsembleMetric:
    """
   Abstract Class for ensemble metrics.
   """

    def get_name(self):
        """
        Returns the name of the metric (as string)
        """
        pass

    def compute_diversity(self, clf_predictions, test_y):
        """
        Abstract method to compute diversity metric
        :param clf_predictions: predictions of clfs in the ensemble
        :param test_y: reference test labels
        :return: a float value
        """
        pass


class QStatMetric(EnsembleMetric):
    """

    """

    def __init__(self):
        """
        Constructor Method
        """
        return

    def compute_diversity(self, clf_predictions, test_y):
        """
        Abstract method to compute diversity metric
        :param clf_predictions: predictions of clfs in the ensemble
        :param test_y: reference test labels
        :return: a float value
        """
        n_clfs = clf_predictions.shape[1]
        qd = QStatDiversity()
        diversities = [qd.compute_diversity(clf_predictions[:, i], clf_predictions[:, j], test_y)
                       for i in range(0, n_clfs - 1) for j in range(i + 1, n_clfs)]
        return 2 * sum(diversities) / (n_clfs * (n_clfs - 1))

    def get_name(self):
        return 'QStat'


class SigmaMetric(EnsembleMetric):
    """

    """

    def __init__(self):
        """
        Constructor Method
        """
        return

    def compute_diversity(self, clf_predictions, test_y):
        """
        Abstract method to compute diversity metric
        :param clf_predictions: predictions of clfs in the ensemble
        :param test_y: reference test labels
        :return: a float value
        """
        n_clfs = clf_predictions.shape[1]
        qd = SigmaDiversity()
        diversities = [qd.compute_diversity(clf_predictions[:, i], clf_predictions[:, j], test_y)
                       for i in range(0, n_clfs - 1) for j in range(i + 1, n_clfs)]
        return 2 * sum(diversities) / (n_clfs * (n_clfs - 1))

    def get_name(self):
        return 'Sigma'


class CoupleDisagreementMetric(EnsembleMetric):
    """

    """

    def __init__(self):
        """
        Constructor Method
        """
        return

    def compute_diversity(self, clf_predictions, test_y):
        """
        Abstract method to compute diversity metric
        :param clf_predictions: predictions of clfs in the ensemble
        :param test_y: reference test labels
        :return: a float value
        """
        n_clfs = clf_predictions.shape[1]
        qd = Disagreement()
        diversities = [qd.compute_diversity(clf_predictions[:, i], clf_predictions[:, j], test_y)
                       for i in range(0, n_clfs - 1) for j in range(i + 1, n_clfs)]
        return numpy.mean(diversities)

    def get_name(self):
        return 'CoupleDisagreement'


class DisagreementMetric(EnsembleMetric):
    """

    """

    def __init__(self):
        """
        Constructor Method
        """
        return

    def compute_diversity(self, clf_predictions, test_y):
        """
        Abstract method to compute diversity metric
        :param clf_predictions: predictions of clfs in the ensemble
        :param test_y: reference test labels
        :return: a float value
        """
        clf_hits = numpy.asarray(
            [(clf_predictions[:, i] == test_y) for i in range(0, clf_predictions.shape[1])]).transpose()
        agreements = [numpy.all(clf_hits[i, :] == clf_hits[i, 0]) for i in range(0, clf_predictions.shape[0])]
        return len(test_y) - sum(agreements)

    def get_name(self):
        return 'Disagreement'


class SharedFaultMetric(EnsembleMetric):
    """

    """

    def __init__(self):
        """
        Constructor Method
        """
        return

    def compute_diversity(self, clf_predictions, test_y):
        """
        Abstract method to compute diversity metric
        :param clf_predictions: predictions of clfs in the ensemble
        :param test_y: reference test labels
        :return: a float value
        """
        comp_array = [False for i in range(0, clf_predictions.shape[1])]
        clf_hits = numpy.asarray(
            [(clf_predictions[:, i] == test_y) for i in range(0, clf_predictions.shape[1])]).transpose()
        shared_faults = numpy.asarray(
            [numpy.all(clf_hits[i, :] == comp_array) for i in range(0, clf_predictions.shape[0])])
        return sum(shared_faults)

    def get_name(self):
        return 'SharedFault'
