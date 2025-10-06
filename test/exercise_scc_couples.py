import os

import numpy
import pandas

from omelet.utils.classifier_utils import compute_omission_metrics

FCC_CSV_FOLDER = "output_folder"
SCORES_FILE = "test_couple_rb_002.csv"
OUTPUT_FOLDER = "matrixes"

REJECT_TAG = -1

# ----------------------- MAIN ROUTINE ---------------------
# This script replicates experiments done for testing the robustness of confidence ensembles
if __name__ == '__main__':

    with open(SCORES_FILE, 'w') as file_handler:
        file_handler.write("dataset,alr,scc1,scc2,val_gain,val_drop,val_aw,val_phi,val_ew,test_aw,test_phi,test_ew\n")

    # Iterating over datasets
    for dataset_file in os.listdir(FCC_CSV_FOLDER):
        # if file is a CSV, it is assumed to be a dataset to be processed
        if dataset_file.endswith("@TEST.csv"):
            dataset_name = dataset_file.split("@")[0]
            alr = dataset_file.split("@")[1].replace("ALR=", "")

            # load validation and test csv
            val_res_df = pandas.read_csv(os.path.join(FCC_CSV_FOLDER, dataset_name + "@ALR=" + alr + "@VALIDATION.csv"))
            test_res_df = pandas.read_csv(os.path.join(FCC_CSV_FOLDER, dataset_name + "@ALR=" + alr + "@TEST.csv"))

            # derive sccs that meet the ALR constraint
            sccs = []
            for col_name in val_res_df.columns:
                if "feature_" not in col_name and col_name != "true_label" and col_name.endswith("_pred"):
                    sccs.append(col_name)
            if len(sccs) > 0:
                # Compute Gain and Drop
                gain = numpy.zeros((len(sccs), len(sccs)))
                drop = numpy.zeros((len(sccs), len(sccs)))
                for i in range(0, len(sccs)):
                    first = val_res_df[sccs[i]]
                    first_t = test_res_df[sccs[i]]
                    for j in range(0, len(sccs)):
                        second = val_res_df[sccs[j]]
                        second_t = test_res_df[sccs[j]]
                        gain[i, j] = sum((first == -1) * (second == val_res_df["true_label"])) / len(first)
                        drop[i, j] = sum((first == -1) * (second != val_res_df["true_label"]) * (second != -1)) / len(first)
                        rb_predict = numpy.asarray([first[i] if first[i] != -1 else second[i] for i in range(0, len(first))])
                        val_fcc_metrics = compute_omission_metrics(val_res_df["true_label"], rb_predict, reject_tag=REJECT_TAG)
                        rb_predict_t = numpy.asarray([first_t[i] if first_t[i] != -1 else second_t[i] for i in range(0, len(first_t))])
                        test_fcc_metrics = compute_omission_metrics(test_res_df["true_label"], rb_predict_t, reject_tag=REJECT_TAG)
                        with open(SCORES_FILE, 'a') as file_handler:
                            file_handler.write(dataset_name + "," + str(alr) + "," + sccs[i] + "," + sccs[j] + "," +
                                               str(gain[i, j]) + "," + str(drop[i, j]) + "," + str(val_fcc_metrics["aw"]) +
                                               "," + str(val_fcc_metrics["phi"]) + "," + str(val_fcc_metrics["ew"]) + "," +
                                               str(test_fcc_metrics["aw"]) + "," + str(test_fcc_metrics["phi"]) + "," +
                                               str(test_fcc_metrics["ew"]) + "\n")

                gain_df = pandas.DataFrame(data=gain, columns=sccs, index=sccs)
                gain_df.to_csv(os.path.join(OUTPUT_FOLDER, dataset_file.replace("@TEST", "@GAIN")))
                drop_df = pandas.DataFrame(data=drop, columns=sccs, index=sccs)
                drop_df.to_csv(os.path.join(OUTPUT_FOLDER, dataset_file.replace("@TEST", "@DROP")))

            else:
                print("no candidate SCCs for this dataset [%s]" % dataset_name)
