import json
import re
from datetime import datetime
from datetime import timedelta

import numpy as np
from sklearn import metrics


class LogsAnalyzer:
    @staticmethod
    def parse_train_log(file_name):
        train_log_re = re.compile(".* INFO Train .*")
        eval_log_re = re.compile(".* INFO Eval .*")
        rmse_re = re.compile(".* RMSE = ?([0-9]*[.][0-9]+)")
        date_time = re.compile("(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})")

        train_results = []
        eval_results = []

        with open(file_name, "r") as logs_file:
            first_line = None
            last_line = None

            for line in logs_file:
                search_res = rmse_re.search(line)
                if search_res is None:
                    continue

                rmse = float(search_res.groups()[0])

                if train_log_re.match(line):
                    train_results.append(rmse)
                elif eval_log_re.match(line):
                    eval_results.append(rmse)

                if first_line is None:
                    first_line = line

                last_line = line

            start_time = datetime.strptime(date_time.match(first_line).group(), "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(date_time.match(last_line).group(), "%Y-%m-%d %H:%M:%S")
            time_diff = end_time - start_time

        return train_results, eval_results, time_diff

    @staticmethod
    def parse_train_logs(file_names):
        train_results = []
        eval_results = []
        training_time = timedelta(0)

        for file_name in file_names:
            train, eval, time_diff = LogsAnalyzer.parse_train_log(file_name)
            train_results.extend(train)
            eval_results.extend(eval)
            training_time += time_diff

        return train_results, eval_results, training_time

    @staticmethod
    def parse_score(score_file_name):
        iteration = 0
        scores = []
        scores_actual = []
        scores_predicted= []

        with open(score_file_name, "r") as json_file:
            for line in json_file:
                iteration = iteration + 1

                values = json.loads(line)
                actual = float(values["actual"])
                predicted = float(values["predicted"])
                scores.append({"iteration": iteration, "actual": actual, "predicted": predicted})

                scores_actual.append(actual)
                scores_predicted.append(predicted)

        return scores, np.asarray(scores_actual), np.asarray(scores_predicted)

    @staticmethod
    def get_optimal_threshold(actual, scores, rate1, rate2):
        filter = np.where(np.logical_and(rate1 <= actual, actual <= rate2))
        fpr, tpr, thresholds = metrics.roc_curve(actual[filter], scores[filter], pos_label=rate2)

        print("AUC {} {}: {}".format(rate1, rate2, metrics.auc(fpr, tpr)))

        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        return {"threshold": optimal_threshold, "fpr": fpr, "tpr": tpr, "thresholds": thresholds}

    @staticmethod
    def get_optimal_threshold_binary(actual, scores):
        filter = np.where(actual != 3.0)
        fpr, tpr, thresholds = metrics.roc_curve(3.0 < actual[filter], scores[filter])

        print("AUC polarity: {}".format(metrics.auc(fpr, tpr)))

        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        return {"threshold": optimal_threshold, "fpr": fpr, "tpr": tpr, "thresholds": thresholds}

    @staticmethod
    def get_compared_scores(score_file_name):
        scores, actual_values, pred_scores = LogsAnalyzer.parse_score(score_file_name)

        scores_actual = []
        scores_predict = []

        scores_actual_binary = []
        scores_predict_binary = []

        roc_results = [LogsAnalyzer.get_optimal_threshold(actual_values, pred_scores, 1.0, 2.0),
                       LogsAnalyzer.get_optimal_threshold(actual_values, pred_scores, 2.0, 3.0),
                       LogsAnalyzer.get_optimal_threshold(actual_values, pred_scores, 3.0, 4.0),
                       LogsAnalyzer.get_optimal_threshold(actual_values, pred_scores, 4.0, 5.0)]

        roc_binary = LogsAnalyzer.get_optimal_threshold_binary(actual_values, pred_scores)

        for s in scores:
            actual = s["actual"]
            predicted = s["predicted"]
            s["predicted_positive"] = 3.0 < predicted

            if predicted <= roc_results[0]["threshold"]:
                predicted_rate = 1.0
            elif predicted <= roc_results[1]["threshold"]:
                predicted_rate = 2.0
            elif predicted <= roc_results[2]["threshold"]:
                predicted_rate = 3.0
            elif predicted <= roc_results[3]["threshold"]:
                predicted_rate = 4.0
            else:
                predicted_rate = 5.0

            s["predicted_rate"] = predicted_rate
            s["prediction_right"] = predicted_rate == actual

            scores_actual.append(actual)
            scores_predict.append(predicted_rate)

            if actual != 3.0:
                scores_actual_binary.append(3.0 < actual)
                scores_predict_binary.append(roc_binary["threshold"] < predicted)

        return scores, scores_actual, scores_predict, scores_actual_binary, \
               scores_predict_binary, roc_results, roc_binary
