"""Balance and capacity selection, followed by refitting on matured labels."""
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score

from .tree_baseline import _xgb_params

BALANCE_GRID = (1.0, 5.0, 20.0)
CAPACITY_GRID = ((8, 500), (10, 500), (8, 1000), (10, 1000))


def standard_weight(labels, config):
    positives = float(max(1.0, np.asarray(labels, dtype=np.float32).sum()))
    return min(config['tree_max_scale_pos_weight'],
               max(1.0, (len(labels) - positives) / positives))


def default_recipe(labels, config):
    return standard_weight(labels, config), CAPACITY_GRID[0]


def _fit(train, config, seed, recipe, valid=None):
    weight, (depth, rounds) = recipe
    params = _xgb_params(dict(config, seed=seed, tree_max_depth=depth),
                         config['_use_gpu'])
    params['scale_pos_weight'] = weight
    options = {} if valid is None else {
        'evals': [(valid, 'validation')],
        'early_stopping_rounds': config['framework_early_stopping_rounds'],
    }
    return xgb.train(params, train, num_boost_round=rounds, verbose_eval=False,
                     **options)


def backbone_recipe(train_x, train_y, valid_x, valid_y, config, seed):
    recipe = default_recipe(train_y, config)
    _fit(xgb.DMatrix(train_x, label=train_y), config, seed, recipe,
         xgb.DMatrix(valid_x, label=valid_y))
    return recipe


def select_recipe(train_x, train_y, valid_x, valid_y, config, seed):
    if int(valid_y.sum()) < config['min_calibration_positives']:
        return backbone_recipe(train_x, train_y, valid_x, valid_y, config, seed)

    train = xgb.DMatrix(train_x, label=train_y)
    valid = xgb.DMatrix(valid_x, label=valid_y)

    def validation_ap(recipe):
        booster = _fit(train, config, seed, recipe, valid)
        # Use all retained trees, not just the best early-stopping iteration.
        return average_precision_score(valid_y, booster.predict(valid))

    best_weight, best_capacity = BALANCE_GRID[0], CAPACITY_GRID[0]
    best_ap = float('-inf')
    for weight in BALANCE_GRID:
        ap = validation_ap((weight, best_capacity))
        if ap > best_ap:
            best_weight, best_ap = weight, ap
    for capacity in CAPACITY_GRID[1:]:
        ap = validation_ap((best_weight, capacity))
        if ap > best_ap:
            best_capacity, best_ap = capacity, ap
    return best_weight, best_capacity


def fit_static(train_x, train_y, valid_x, valid_y, test_x, config, seed, recipe):
    train = xgb.DMatrix(np.concatenate([train_x, valid_x]),
                        label=np.concatenate([train_y, valid_y]))
    booster = _fit(train, config, seed, recipe)
    return booster.predict(xgb.DMatrix(test_x))
