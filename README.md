# OMELET
Framework to support the safe classificatiOn via enseMblEs of SeLf controllEd componenTs

Anonymized and Supporting repository for the ICSE Submission #2434 (Research Track - II Cycle)

## Description

The reporitory contains code used to generate experimental results presented in the submitted paper. At its current state of the implementation, it allows for experimenting with many rejection strategies and classifiers. It is tested only with tabular data, but it is theoretically applicable also to other classification problems.

## Main Dependencies
OMELET needs the following libraries:
- <a href="https://numpy.org/">NumPy</a>
- <a href="https://scipy.org/">SciPy</a>
- <a href="https://pandas.pydata.org/">Pandas</a>
- <a href="https://scikit-learn.org/stable/">Scikit-Learn</a>
- <a href="https://xgboost.readthedocs.io/en/stable/python/python_intro.html">xgboost</a>
- <a href="https://pypi.org/project/confidence-ensembles/">confidence-ensembles</a>
- <a href="https://github.com/yzhao062/pyod">sprout-ml</a> (for rejection strategies, called "Uncertainty Measures" in the library)

## Usage



## Data Availability
Data used within the experimental section of the paper is not ours, thus we cannot share it directly (but it is references in the paper). To allow reviewers cross-checking our results, we provide a password-protected ZIPfile that contains the preprocessed datasets we used in our experiments. The key to open the ZIPfile is the ID of the paper submission.

The ZIPfile is in icse/input_folder_icse path of the repository


