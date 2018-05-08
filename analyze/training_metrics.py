from analyze.logs_analyzer import LogsAnalyzer
import matplotlib.pyplot as plt


def _generate_training_rmse_chart(plot_name, log_files):
    train_logs, eval_logs= LogsAnalyzer.parse_train_logs(log_files)
    epochs = range(1, len(eval_logs) + 1)

    fig, ax = plt.subplots()
    ax.plot(epochs, eval_logs)

    ax.set(xlabel='epoch', ylabel='RMSE',
           title='{} Training: Root Mean Square Error'.format(plot_name))
    ax.grid()

    fig.savefig("./output/plots/training/{}.png".format(plot_name))
    plt.show()


def build_training_rmse_charts():
    # standard_run
    plot_name = "Standard"
    log_files = ["data/models/logs/standard_run_1/train.log",
                 "data/models/logs/standard_run_1_2/train.log",
                 "data/models/logs/standard_run_1_3/train.log"]
    _generate_training_rmse_chart(plot_name, log_files)


    # standard_group_run
    plot_name = "Standard Group"
    log_files = ["data/models/logs/standard_group_run_1/train.log",
                 "data/models/logs/standard_group_run_1_2/train.log"]
    _generate_training_rmse_chart(plot_name, log_files)

    # # ascii_run
    plot_name = "ASCII"
    log_files = ["data/models/logs/ascii_run_1/train.log",
                 "data/models/logs/ascii_run_1_2/train.log",
                 "data/models/logs/ascii_run_1_3/train.log",
                 "data/models/logs/ascii_run_1_4/train.log"]
    _generate_training_rmse_chart(plot_name, log_files)

    # ascii_group_run
    plot_name = "ASCII Group"
    log_files = ["data/models/logs/ascii_group_run_1/train.log",
                 "data/models/logs/ascii_group_run_1_2/train.log",
                 "data/models/logs/ascii_group_run_1_3/train.log",
                 "data/models/logs/ascii_group_run_1_4/train.log"]
    _generate_training_rmse_chart(plot_name, log_files)
