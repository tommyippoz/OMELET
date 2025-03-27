import numpy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array

from omelet.utils.classifier_utils import get_classifier_name


class AbstractClassifier(BaseEstimator, ClassifierMixin):
    """
    Basic Abstract Class for Classifiers.
    Abstract methods are only the fit_classifier, with many degrees of freedom in implementing them.
    Wraps implementations from different frameworks (if needed), sklearn and many deep learning utilities
    """

    def __init__(self):
        """
        Constructor of a generic Classifier
        :param clf: algorithm to be used as Classifier
        """
        self._estimator_type = "classifier"
        self.feature_importances_ = None
        self.X_ = None
        self.y_ = None
        self.is_fit = False

    def fit(self, X, y=None):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit + other data
        if y is not None:
            self.classes_ = numpy.unique(y)
        else:
            self.classes_ = [0, 1]

        # Train clf
        self.fit_classifier(X, y)
        self.is_fit = True
        # Return the classifier
        return self

    def fit_classifier(self, X, y=None):
        """
        To be overridden
        :param X: train data
        :param y: train labels
        :return:
        """
        pass

    def predict(self, X):
        """
        Method to compute predict of a classifier
        :return: array of predicted class
        """
        probas = self.predict_proba(X)
        return self.classes_[numpy.argmax(probas, axis=1)]

    def predict_proba(self, X):
        """
        Method to compute probabilities of predicted classes
        :return: array of probabilities for each classes
        """

        # Check if fit has been called
        check_is_fitted(self)
        X = check_array(X)

        return self.classifier_predict_proba(X)

    def classifier_predict_proba(self, X):
        """
        To be overridden
        :param X: test data
        :return:
        """
        return None

    def predict_confidence(self, X):
        """
        Method to compute confidence in the predicted class
        :return: max probability as default
        """
        probas = self.predict_proba(X)
        return numpy.max(probas, axis=1)

    def classifier_name(self):
        """
        Returns the name of the classifier (as string)
        """
        return get_classifier_name(self.clf)