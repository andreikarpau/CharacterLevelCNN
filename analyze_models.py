import analyze.training_metrics as training
import analyze.eval_metrics as eval


training.build_training_rmse_charts()
eval.print_confusion_matrices()