# OMELET
Framework to support the safe classificatiOn via enseMblEs of SeLf controllEd componenTs

Anonymized and Supporting repository for the Paper Submission

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
There are two main scripts
- icse/exercise_single_scc.py: this allows for using many classifiers, ALRs and rejection strategies to create SCCs that are exercised on different datasets. Performance measures of these SCCs are saved in a CSV file, and additional support files are saved in the icse/output_folder
- the files in icse/output_folder (i.e., predictions of individual SCCs) are used as input for the icse/exercise_scc_couples.py, which creates couples of SCCs and exercised them on different datasets, computing performance metrics (including gain and drop quantities)

## Plots and Data in the Paper
The ouputs of the two scripts above were copy pasted into "singles" and ""couples" tabs of the excel file in icse/Excel_files and used to extract tables and plots as detailed in the XLSX file (see the README tab within the file)

## Data Availability
Data used within the experimental section of the paper is not ours, thus we cannot share it directly (but it is references in the paper). To allow reviewers cross-checking our results, we provide a password-protected ZIPfile that contains the preprocessed datasets we used in our experiments. The key to open the ZIPfile is the ID of the paper submission.

The ZIPfile is in icse/input_folder_icse path of the repository
