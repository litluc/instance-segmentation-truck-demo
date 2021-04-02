## Scripts

| File name | Purpose |
|---|---|
| [model_train.py](model_train.py) | vanilla training script that sets the configs and kick off training|
| [model_train_custom_aug.py](model_train_custom_aug.py) | training script that adds custom data augmentation to training data |
|[model_eval.py](model_eval.py) | model evaluation script that scores and evaluates a test set |

## Utility modules
| File name | Purpose |
|---|---|
| [config.py](config.py) | defines project level settings, e.g. dir names, datasets |
| [data_utils.py](data_utils.py) | functions for data and data augmentations |
| [custom_trainers.py](custom_trainers.py) | customized Trainers for model training|