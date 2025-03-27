import numpy
import numpy as np


# ALL

def cost_based_threshold(k):
    t = (k) / (k + 1)
    return t


class CostMetric:

    def __init__(self, logits_val, y_val, logits_test, y_test):
        self.logits_val = logits_val
        self.y_val = y_val
        self.logits_test = logits_test
        self.y_test = y_test

    def cost_based_analysis(self, Vr, Vc, Vw, confT_list):
        pass


class BinaryCostMetric(CostMetric):

    def __init__(self, Vw_fp, Vw_fn, Vc_tp, Vc_tn, Vr):
        super().__init__()
        self.V_fp = Vw_fp
        self.V_fn = Vw_fn
        self.Vr = Vr
        self.V_tp = Vc_tp
        self.V_tn = Vc_tn

    def calculate_value(self, y_proba, y_true, t_fp, t_fn):
        y_label = numpy.argmax(y_proba, axis=0)
        prob_positive = y_proba[:, 1]
        prob_negative = y_proba[:, 0]

        y_pred_pos = np.full(prob_positive.shape[0], -1)
        y_pred_neg = np.full(prob_negative.shape[0], -1)

        y_pred_pos[prob_positive > t_fp] = 1
        y_pred_neg[prob_negative > t_fn] = 0

        max_prob_indices = list(np.argmax(y_proba, axis=1))

        y_pred_classic = np.argmax(y_proba, axis=1)
        y_pred = np.array \
            ([y_pred_neg[i] if max_prob_indices[i] == 0 else y_pred_pos[i] for i in range(len(max_prob_indices))])

        # now lets compute the actual value of each prediction
        value_vector = np.full(y_pred.shape[0], self.Vc)
        false_positives_idx = (y_pred == 1) & (y_label == 0)
        false_negatives_idx = (y_pred == 0) & (y_label == 1)

        value_vector[false_positives_idx] = self.V_fp
        value_vector[false_negatives_idx] = self.V_fn

        # loss due to asking humans
        value_vector[y_pred == -1] = self.Vr

        value = np.sum(value_vector) / len(y_label)

        numOfRejectedSamples = np.count_nonzero(y_pred == -1)
        numOfWrongPredictions = np.count_nonzero((y_pred != y_label) & (y_pred != -1))
        numRejectedFp = np.count_nonzero((y_pred_classic == 1) & (y_pred == -1) & (y_label == 0))
        numRejectedFn = np.count_nonzero((y_pred_classic == 0) & (y_pred == -1) & (y_label == 1))
        numRejectedCorrect = numOfRejectedSamples - numRejectedFn - numRejectedFp
        return value, numOfRejectedSamples, numOfWrongPredictions, numRejectedCorrect, numRejectedFp, numRejectedFn

    def find_optimum_confidence_threshold(self, y_proba, y_true, t_list):
        cost_list = {}

        for t_fp in t_list:
            for t_fn in t_list:
                # here we define K = fn_c_norm, change it based on task.
                value = self.calculate_value(y_proba, y_true, t_fp, t_fn)
                cost_list["{}_{}".format(t_fp, t_fn)] = value
        # find t values with maximum value
        maxValue = max(cost_list.values())
        optTList = [[float(k.split('_')[0]), float(k.split('_')[1])] for k, v in cost_list.items() if v == maxValue]

        return optTList[0], cost_list

    def cost_based_analysis(self, y_proba, y_true, confT_list):

        k_fp = (-1) * (self.V_fp / self.Vc)
        k_fn = (-1) * (self.V_fn / self.Vc)
        t_fp = cost_based_threshold(k_fp)
        t_fn = cost_based_threshold(k_fn)

        value_test, rej_test, wrong_test, rej_ok, rej_fp, rej_fn = \
            self.calculate_value(y_proba, y_true, t_fp, t_fn)
        t_optimal, cost_list = self.find_optimum_confidence_threshold(y_proba, y_true, confT_list)
        value_test_opt, rej_test_opt, wrong_test_opt, rej_ok_opt, rej_fp_opt, rej_fn_opt = \
            self.calculate_value(y_proba, y_true, t_optimal[0], t_optimal[1])

        return {'test_size': y_proba.shape[0],
                'cost_rejection': self.Vr,
                'cost_hit': self.Vc,
                'cost_misc_fp': self.V_fp,
                'cost_misc_fn': self.V_fn,
                'k_fp': k_fp,
                'k_fn': k_fn,
                't_fp': t_fp,
                't_fn': t_fn,
                'value': value_test,
                'rejected': rej_test,
                'rej_corr': rej_ok,
                'rej_fp': rej_fp,
                'rej_fn': rej_fn,
                'wrong': wrong_test,
                'correct': y_proba.shape[0] - rej_test - wrong_test,
                't_opt_fp': t_optimal[0],
                't_opt_fn': t_optimal[1],
                'value_opt': value_test_opt,
                'rejected_opt': rej_test_opt,
                'rej_corr_opt': rej_ok_opt,
                'rej_fp_opt': rej_fp_opt,
                'rej_fn_opt': rej_fn_opt,
                'wrong_opt': wrong_test_opt,
                'correct_opt': y_proba.shape[0] - rej_test_opt - wrong_test_opt}


# MULTI


class MultiClassCostMetric(CostMetric):

    def __init__(self, logits_val, y_val, logits_test, y_test):
        super().__init__(logits_val, y_val, logits_test, y_test)

    def calculate_value(self, y_hat_proba, y, t, Vr, Vc, Vw):
        y_pred_classic = np.argmax(y_hat_proba, axis=1)
        y_pred = np.array([np.where(l == np.amax(l))[0][0] if (np.amax(l) > t) else -1 for l in y_hat_proba])

        # now lets compute the actual value of each prediction

        value_vector = np.full(y_pred.shape[0], Vc)

        value_vector[(y_pred != y) & (y_pred != -1)] = Vw

        # loss due to asking humans
        value_vector[y_pred == -1] = Vr
        value = np.sum(value_vector) / len(y)

        numOfRejectedSamples = np.count_nonzero(y_pred == -1)
        numOfWrongPredictions = np.count_nonzero((y_pred != y) & (y_pred != -1))
        numRejectedMisc = np.count_nonzero((y_pred_classic != y) & (y_pred == -1))
        numRejectedCorrect = numOfRejectedSamples - numRejectedMisc
        return value, numOfRejectedSamples, numOfWrongPredictions, numRejectedCorrect, numRejectedMisc

    def find_optimum_confidence_threshold(self, y_hat_proba, y, t_list, Vr, Vc, Vw):
        cost_list = {}

        for t in t_list:
            value, _, __, ___, ____ = self.calculate_value(y_hat_proba, y, t, Vr, Vc, Vw)
            cost_list["{}".format(t)] = value
        # find t values with maximum value
        maxValue = max(cost_list.values())
        optTList = [float(k) for k, v in cost_list.items() if v == maxValue]
        # pick the one with the lowest confidence
        optimumT = min(optTList)

        return optimumT, cost_list

    # cost based calibration analysis
    def cost_based_analysis(self, Vr, Vc, Vw, confT_list):

        k = (-1) * (Vw / Vc)
        t = cost_based_threshold(k)
        value_test, rej_test, wrong_test, rej_ok, rej_misc = self.calculate_value(self.logits_test, self.y_test, t, Vr, Vc, Vw)

        t_optimal, cost_list = self.find_optimum_confidence_threshold(self.logits_val, self.y_val, confT_list, Vr,
                                                                      Vc, Vw)
        value_test_opt, rej_test_opt, wrong_test_opt, rej_ok_opt, rej_misc_opt = self.calculate_value(self.logits_test, self.y_test,
                                                                            t_optimal, Vr, Vc, Vw)

        return {'test_size': len(self.y_test),
                'cost_rejection': Vr,
                'cost_hit': Vc,
                'cost_misc': Vw,
                'k': k,
                't': t,
                'value': value_test,
                'rejected': rej_test,
                'rej_corr': rej_ok,
                'rej_misc': rej_misc,
                'wrong': wrong_test,
                'correct': len(self.y_test) - rej_test - wrong_test,
                't_opt': t_optimal,
                'value_opt': value_test_opt,
                'rejected_opt': rej_test_opt,
                'rej_corr_opt': rej_ok_opt,
                'rej_misc_opt': rej_misc_opt,
                'wrong_opt': wrong_test_opt,
                'correct_opt': len(self.y_test) - rej_test_opt - wrong_test_opt}
