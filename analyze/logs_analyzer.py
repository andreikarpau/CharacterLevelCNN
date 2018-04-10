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
                rmse = float(rmse_re.search(line).groups()[0])

                if train_log_re.match(line):
                    print("train {}".format(rmse))
                elif eval_log_re.match(line):
                    print("eval {}".format(rmse))
