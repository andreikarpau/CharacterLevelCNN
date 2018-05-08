from pandas_ml import ConfusionMatrix
from analyze.logs_analyzer import LogsAnalyzer


def _print_confusion_matrix(name, score_file_name):
    scores, s_actual, s_predict, s_actual_bin, s_predict_bin = LogsAnalyzer.get_compared_scores(score_file_name)

    conf = ConfusionMatrix(s_actual, s_predict)
    conf_bin = ConfusionMatrix(s_actual_bin, s_predict_bin)

    print("----------------{}---------------\n".format(name))
    print("Score classification: ")
    print("overall accuracy = {}\n".format(conf.stats_overall["Accuracy"]))
    print(conf)
    print("\nBinary classification: ")
    print("accuracy = {}\n".format(conf_bin.ACC))
    print(conf_bin)
    print("\n")


def print_confusion_matrices():
    # standard_run
    name = "Standard"
    score_file_name = "data/models/eval/standard_run_1_3_e13/score.json"
    _print_confusion_matrix(name, score_file_name)

    # standard_group_run
    name = "Standard Group"
    score_file_name = "data/models/eval/standard_group_run_1_2_e21/score.json"
    _print_confusion_matrix(name, score_file_name)

    # # ascii_run
    name = "ASCII"
    score_file_name = "data/models/eval/ascii_run_1_2_e49/score.json"
    _print_confusion_matrix(name, score_file_name)

    # ascii_group_run
    name = "ASCII Group"
    score_file_name = "data/models/eval/ascii_group_run_1_3_e49/score.json"
    _print_confusion_matrix(name, score_file_name)
