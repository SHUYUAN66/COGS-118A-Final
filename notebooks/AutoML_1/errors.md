## Error for 3_Default_Xgboost

Stop training after the first fold. Time needed to train on the first fold 674.0 seconds. The time estimate for training on all folds is larger than total_time_limit.
Traceback (most recent call last):
  File "/Users/anthea-mac/opt/anaconda3/envs/rna/lib/python3.8/site-packages/supervised/base_automl.py", line 900, in _fit
    trained = self.train_model(params)
  File "/Users/anthea-mac/opt/anaconda3/envs/rna/lib/python3.8/site-packages/supervised/base_automl.py", line 300, in train_model
    mf.train(model_path)
  File "/Users/anthea-mac/opt/anaconda3/envs/rna/lib/python3.8/site-packages/supervised/model_framework.py", line 172, in train
    self.callbacks.on_learner_train_end()
  File "/Users/anthea-mac/opt/anaconda3/envs/rna/lib/python3.8/site-packages/supervised/callbacks/callback_list.py", line 15, in on_learner_train_end
    cb.on_learner_train_end(logs)
  File "/Users/anthea-mac/opt/anaconda3/envs/rna/lib/python3.8/site-packages/supervised/callbacks/total_time_constraint.py", line 43, in on_learner_train_end
    raise AutoMLException(
supervised.exceptions.AutoMLException: Stop training after the first fold. Time needed to train on the first fold 674.0 seconds. The time estimate for training on all folds is larger than total_time_limit.


Please set a GitHub issue with above error message at: https://github.com/mljar/mljar-supervised/issues/new

