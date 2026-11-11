"""External detectors under the same label-maturity and refitting protocol."""
import numpy as np

from .tree_baseline import _xgb_params
from .tree_framework import standard_weight

METHODS = ('lgb', 'catboost', 'xgb_ros', 'xgb_rus')


def _resample(x, y, seed, oversample):
    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    target = 10 * len(positives)
    if oversample:
        extra = rng.choice(positives, size=target - len(positives), replace=True)
        keep = np.concatenate([np.arange(len(y)), extra])
    else:
        chosen = rng.choice(negatives, size=min(target, len(negatives)), replace=False)
        keep = np.concatenate([positives, chosen])
    keep.sort()
    return x[keep], y[keep]


def _fit_lgb(x, y, valid_x, valid_y, test_x, config, seed, rounds=None):
    import lightgbm as lgb

    params = {
        'objective': 'binary',
        'metric': 'average_precision',
        'learning_rate': config['tree_learning_rate'],
        'num_leaves': 63,
        'min_child_samples': 20,
        'feature_fraction': config['tree_colsample'],
        'bagging_fraction': config['tree_subsample'],
        'bagging_freq': 1,
        'lambda_l2': config['tree_l2'],
        'scale_pos_weight': standard_weight(y, config),
        'seed': seed,
        'verbosity': -1,
        'num_threads': 32,
    }
    train = lgb.Dataset(x, label=y)
    if rounds is None and len(valid_y):
        valid = lgb.Dataset(valid_x, label=valid_y, reference=train)
        model = lgb.train(params, train, num_boost_round=config['tree_rounds'],
                          valid_sets=[valid], callbacks=[lgb.early_stopping(
                              config['tree_early_stopping_rounds'], verbose=False)])
        return model.best_iteration or config['tree_rounds']
    model = lgb.train(params, train, num_boost_round=rounds or config['tree_rounds'])
    return model.predict(test_x) if rounds is not None else config['tree_rounds']


def _fit_catboost(x, y, valid_x, valid_y, test_x, config, seed, rounds=None):
    from catboost import CatBoostClassifier

    options = {}
    if rounds is None and len(valid_y):
        options['early_stopping_rounds'] = config['tree_early_stopping_rounds']
    model = CatBoostClassifier(
        depth=config['tree_max_depth'], learning_rate=config['tree_learning_rate'],
        l2_leaf_reg=config['tree_l2'], scale_pos_weight=standard_weight(y, config),
        eval_metric='PRAUC', random_seed=seed, verbose=False, allow_writing_files=False,
        task_type='GPU' if config['_use_gpu'] else 'CPU', bootstrap_type='Bernoulli',
        subsample=config['tree_subsample'], iterations=rounds or config['tree_rounds'],
        **options,
    )
    if options:
        model.fit(x, y, eval_set=(valid_x, valid_y))
        return model.get_best_iteration() or config['tree_rounds']
    model.fit(x, y)
    return model.predict_proba(test_x)[:, 1] if rounds is not None else config['tree_rounds']


def _fit_xgb(x, y, valid_x, valid_y, test_x, config, seed, rounds=None):
    import xgboost as xgb

    params = _xgb_params(dict(config, seed=seed), config['_use_gpu'])
    params['scale_pos_weight'] = standard_weight(y, config)
    train = xgb.DMatrix(x, label=y)
    if rounds is None and len(valid_y):
        valid = xgb.DMatrix(valid_x, label=valid_y)
        model = xgb.train(params, train, num_boost_round=config['tree_rounds'],
                          evals=[(valid, 'validation')], verbose_eval=False,
                          early_stopping_rounds=config['tree_early_stopping_rounds'])
        return model.best_iteration + 1
    model = xgb.train(params, train, num_boost_round=rounds or config['tree_rounds'],
                      verbose_eval=False)
    return model.predict(xgb.DMatrix(test_x)) if rounds is not None else config['tree_rounds']


def fit_external(method, train_x, train_y, valid_x, valid_y, test_x, config, seed):
    fit = {'lgb': _fit_lgb, 'catboost': _fit_catboost,
           'xgb_ros': _fit_xgb, 'xgb_rus': _fit_xgb}[method]
    resample = method in ('xgb_ros', 'xgb_rus')
    oversample = method == 'xgb_ros'
    x, y = _resample(train_x, train_y, seed, oversample) if resample else (train_x, train_y)
    best = fit(x, y, valid_x, valid_y, test_x, config, seed)

    x = np.concatenate([train_x, valid_x])
    y = np.concatenate([train_y, valid_y])
    if resample:
        x, y = _resample(x, y, seed, oversample)
    return fit(x, y, valid_x, valid_y, test_x, config, seed, rounds=best)
