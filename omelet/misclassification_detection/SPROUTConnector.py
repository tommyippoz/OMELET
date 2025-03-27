from enum import Enum

from pyod.models.copod import COPOD
from pyod.models.feature_bagging import FeatureBagging
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sprout.SPROUTObject import SPROUTObject
from sprout.classifiers.Classifier import LogisticReg
from xgboost import XGBClassifier


class SPROUTGroup(Enum):
    """
    Supports creation of SPROUTRejection objects
    """
    UM1 = 4
    UM2 = 5
    UM3 = 6
    UM4 = 7
    UM5_XGB = 8
    UM6_NB = 9
    UM6_ST = 10
    UM6_TR = 11
    UM7_DT = 12
    UM8 = 13
    UM9 = 14


def build_sprout_object(x_train, y_train, strategies: list = [], label_names: list = None) -> SPROUTObject:
    sp_obj = SPROUTObject(models_folder="sprout_models")
    if SPROUTGroup.UM1 in strategies:
        # Confidence Intervals
        sp_obj.add_calculator_confidence(x_train=x_train, y_train=y_train, confidence_level=0.9)
    if SPROUTGroup.UM2 in strategies:
        # Maximum Probability
        sp_obj.add_calculator_maxprob()
    if SPROUTGroup.UM3 in strategies:
        # Entropy
        sp_obj.add_calculator_entropy(n_classes=len(label_names) if label_names is not None else 2)
    if SPROUTGroup.UM4 in strategies:
        # Bayesian
        sp_obj.add_calculator_bayes(x_train=x_train, y_train=y_train,
                                    n_classes=len(label_names) if label_names is not None else 2)
    if SPROUTGroup.UM5_XGB in strategies:
        # Combined - Supervised
        sp_obj.add_calculator_combined(classifier=XGBClassifier(n_estimators=100), x_train=x_train,
                                       y_train=y_train,
                                       n_classes=len(label_names) if label_names is not None else 2)
    if SPROUTGroup.UM6_NB in strategies:
        # Multi-Combined
        sp_obj.add_calculator_multicombined(clf_set=[
            Pipeline([("norm", MinMaxScaler()), ("clf", GaussianNB())]),
            Pipeline([("norm", MinMaxScaler()), ("clf", BernoulliNB())]),
            Pipeline([("norm", MinMaxScaler()), ("clf", MultinomialNB())]),
            Pipeline([("norm", MinMaxScaler()), ("clf", ComplementNB())])],
            x_train=x_train, y_train=y_train,
            n_classes=len(label_names) if label_names is not None else 2)
    if SPROUTGroup.UM6_ST in strategies:
        # Multi-Combined
        sp_obj.add_calculator_multicombined(clf_set=[
            Pipeline([("norm", MinMaxScaler()), ("clf", GaussianNB())]),
            LinearDiscriminantAnalysis(),
            LogisticReg()],
            x_train=x_train, y_train=y_train,
            n_classes=len(label_names) if label_names is not None else 2)
    if SPROUTGroup.UM6_TR in strategies:
        # Multi-Combined
        sp_obj.add_calculator_multicombined(clf_set=[
            DecisionTreeClassifier(),
            RandomForestClassifier(n_estimators=10),
            GradientBoostingClassifier(n_estimators=10)],
            x_train=x_train, y_train=y_train,
            n_classes=len(label_names) if label_names is not None else 2)
    if SPROUTGroup.UM7_DT in strategies:
        # Feature Bagging - Supervised
        sp_obj.add_calculator_combined(
            classifier=BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=6)),
            x_train=x_train, y_train=y_train,
            n_classes=len(label_names) if label_names is not None else 2)
    if SPROUTGroup.UM8 in strategies:
        # Neighbour-based uncertainty
        sp_obj.add_calculator_knn_distance(x_train=x_train, k=5)
    if SPROUTGroup.UM9 in strategies:
        # Multi-Combined
        sp_obj.add_calculator_recloss(x_train=x_train, tag='')
    return sp_obj
