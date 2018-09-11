# About data and dataset
data_dir = 'data/'
datasets_params_file = 'data/datasets_params.json'
datasets_log = 'data/datasets.log'
train_name = 'train'
val_name = 'val'
test_name = 'test'
train_factor = 0.70
val_factor = 0.15
test_factor = 0.15
min_count_word = 1
min_count_tag = 1

# An experiment directory
base_model_dir = 'experiments/base_model'

# The configurations, results and logs for one experiment
params_file = 'params.json'
checkpoint_format = '{}.pth.tar'
best_model_file = 'best.pth.tar'
last_model_file = 'last.pth.tar'
best_metrics_format = 'best_metrics_on_{}.json'
last_metrics_format = 'last_metrics_on_{}.json'
best_metrics_on_val_file = 'best_metrics_on_val.json'
last_metrics_on_val_file = 'last_metrics_on_val.json'
best_metrics_on_test_file = 'best_metrics_on_test.json'
last_metrics_on_test_file = 'last_metrics_on_test.json'
train_log = 'train.log'
evaluate_log = 'evaluate.log'

