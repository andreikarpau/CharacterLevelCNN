import json
import numpy as np
from scipy import stats
from analyze.logs_analyzer import LogsAnalyzer
from helpers.preprocess_helper import PreprocessHelper


# ---------------------Bootstrap indexes from test dataset-------------------
# all_indexes = PreprocessHelper.bootstrap_test_indexes()

# with open('analyze/indexes.json', 'w+') as file:
#     json.dump(all_indexes, file)

# ---------------------Analyze logs---------------------
standard_mses_v1 = LogsAnalyzer.parse_eval_bootstap_log("data/models/eval/standard_run_1_5_e48/eval_bootstrap.log")
standard_group_mses_v1 = LogsAnalyzer.parse_eval_bootstap_log("data/models/eval/standard_group_run_1_5_e49/eval_bootstrap.log")
ascii_mses_v1 = LogsAnalyzer.parse_eval_bootstap_log("data/models/eval/ascii_run_1_7_e64/eval_bootstrap.log")
ascii_group_mses_v1 = LogsAnalyzer.parse_eval_bootstap_log("data/models/eval/ascii_group_run_1_7_e48/eval_bootstrap.log")

standard_mses_v2 = LogsAnalyzer.parse_eval_bootstap_log("data/models/eval/standard_run_1_5_e48/eval_bootstrap_v2.log")
standard_group_mses_v2 = LogsAnalyzer.parse_eval_bootstap_log("data/models/eval/standard_group_run_1_5_e49/eval_bootstrap_v2.log")
ascii_mses_v2 = LogsAnalyzer.parse_eval_bootstap_log("data/models/eval/ascii_run_1_7_e64/eval_bootstrap_v2.log")
ascii_group_mses_v2 = LogsAnalyzer.parse_eval_bootstap_log("data/models/eval/ascii_group_run_1_7_e48/eval_bootstrap_v2.log")

standard_mses_v1 = np.asarray(standard_mses_v1)
standard_group_mses_v1 = np.asarray(standard_group_mses_v1)
ascii_mses_v1 = np.asarray(ascii_mses_v1)
ascii_group_mses_v1 = np.asarray(ascii_group_mses_v1)

standard_mses_v2 = np.asarray(standard_mses_v2)
standard_group_mses_v2 = np.asarray(standard_group_mses_v2)
ascii_mses_v2 = np.asarray(ascii_mses_v2)
ascii_group_mses_v2 = np.asarray(ascii_group_mses_v2)

print("Paired Samples T Test (correlated):")
print("Statistics between standard and standard group")
t2_rel, p2_rel = stats.ttest_rel(standard_mses_v1, standard_group_mses_v1)
print("t = {}".format(t2_rel))
print("p = {}".format(p2_rel))
print("mean (Standard) = {}".format(standard_mses_v1.mean()))
print("mean (Standard Group) = {}".format(standard_group_mses_v1.mean()))

print("Statistics between ascii and ascii group")
t2_rel, p2_rel = stats.ttest_rel(ascii_mses_v1, ascii_group_mses_v1)
print("t = {}".format(t2_rel))
print("p = {}".format(p2_rel))
print("mean (ASCII) = {}".format(ascii_mses_v1.mean()))
print("mean (ASCII Group) = {}".format(ascii_group_mses_v1.mean()))

