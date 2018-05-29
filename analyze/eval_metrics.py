from pandas_ml import ConfusionMatrix
from analyze.logs_analyzer import LogsAnalyzer
import matplotlib.pyplot as plt


def _print_roc_curves(roc, roc_bin, name):
    fig, ax = plt.subplots()

    for i in range(0, 4):
        ax.plot(roc[i]["fpr"], roc[i]["tpr"])

    ax.legend(['scores 1, 2', 'scores 2, 3', 'scores 3, 4', 'scores 4, 5'], loc='upper right')
    ax.plot([0, 1], [0, 1], 'gray', dashes=[6, 2])
    ax.set_title("Multiclass ROC {}".format(name))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    fig.savefig("./output/plots/eval/{}_roc.png".format(name))
    plt.show()

    fig, ax = plt.subplots()

    ax.plot(roc_bin["fpr"], roc_bin["tpr"])

    ax.plot([0, 1], [0, 1], 'gray', dashes=[6, 2])
    ax.set_title("Polarity ROC {}".format(name))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    fig.savefig("./output/plots/eval/{}_polarity_roc.png".format(name))
    plt.show()


def _print_confusion_matrix_rocs(name, score_file_name):
    scores, s_actual, s_predict, s_actual_bin, s_predict_bin, roc, roc_bin =\
        LogsAnalyzer.get_compared_scores(score_file_name)

    _print_roc_curves(roc, roc_bin, name)
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
    score_file_name = "data/models/eval/standard_run_1_5_e48/score.json"
    _print_confusion_matrix_rocs(name, score_file_name)

    # standard_group_run
    name = "Standard Group"
    score_file_name = "data/models/eval/standard_group_run_1_5_e49/score.json"
    _print_confusion_matrix_rocs(name, score_file_name)

    # # ascii_run
    name = "ASCII"
    score_file_name = "data/models/eval/ascii_run_1_7_e64/score.json"
    _print_confusion_matrix_rocs(name, score_file_name)

    # ascii_group_run
    name = "ASCII Group"
    score_file_name = "data/models/eval/ascii_group_run_1_7_e48/score.json"
    _print_confusion_matrix_rocs(name, score_file_name)
