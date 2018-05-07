import analyze.training_rmse_plots as rmse_plots


# rmse_plots.build_training_rmse_charts()
from analyze.logs_analyzer import LogsAnalyzer

score_file_name = "data/models/eval/standard_run_1_3_e13/score.json"
scores = LogsAnalyzer.get_compared_scores(score_file_name)

print(scores)
