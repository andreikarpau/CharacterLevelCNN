import json
import re


class LogsAnalyzer:
    @staticmethod
    def parse_train_log(file_name):
        train_log_re = re.compile(".* INFO Train .*")
        eval_log_re = re.compile(".* INFO Eval .*")
        rmse_re = re.compile(".* RMSE = ?([0-9]*[.][0-9]+)")

        train_results = []
        eval_results = []

        with open(file_name, "r") as logs_file:
            for line in logs_file:
                search_res = rmse_re.search(line)
                if search_res is None:
                    continue

                rmse = float(search_res.groups()[0])

                if train_log_re.match(line):
                    train_results.append(rmse)
                elif eval_log_re.match(line):
                    eval_results.append(rmse)

        return train_results, eval_results

    @staticmethod
    def parse_train_logs(file_names):
        train_results = []
        eval_results = []

        for file_name in file_names:
            train, eval = LogsAnalyzer.parse_train_log(file_name)
            train_results.extend(train)
            eval_results.extend(eval)

        return train_results, eval_results

    @staticmethod
    def parse_score(score_file_name):
        iteration = 0
        scores = []

        with open(score_file_name, "r") as json_file:
            for line in json_file:
                iteration = iteration + 1

                values = json.loads(line)
                actual = float(values["actual"])
                predicted = float(values["predicted"])
                scores.append({"iteration": iteration, "actual": actual, "predicted": predicted})

        return scores

    @staticmethod
    def get_compared_scores(score_file_name):
        scores = LogsAnalyzer.parse_score(score_file_name)

        scores_actual = []
        scores_predict = []

        scores_actual_binary = []
        scores_predict_binary = []

        for s in scores:
            actual = s["actual"]
            predicted = s["predicted"]
            s["predicted_positive"] = 3.0 < predicted

            if predicted <= 1.5:
                predicted_rate = 1.0
            elif predicted <= 2.5:
                predicted_rate = 2.0
            elif predicted <= 3.5:
                predicted_rate = 3.0
            elif predicted <= 4.5:
                predicted_rate = 4.0
            else:
                predicted_rate = 5.0

            s["predicted_rate"] = predicted_rate
            s["prediction_right"] = predicted_rate == actual

            scores_actual.append(actual)
            scores_predict.append(predicted_rate)

            if actual != 3.0:
                scores_actual_binary.append(3.0 < actual)
                scores_predict_binary.append(3.0 < predicted)

        return scores, scores_actual, scores_predict, scores_actual_binary, scores_predict_binary






