"""Shared XGBoost training parameters."""


def _xgb_params(config, use_gpu):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'max_depth': int(config['tree_max_depth']),
        'min_child_weight': float(config['tree_min_child_weight']),
        'eta': float(config['tree_learning_rate']),
        'subsample': float(config['tree_subsample']),
        'colsample_bytree': float(config['tree_colsample']),
        'lambda': float(config['tree_l2']),
        'alpha': float(config['tree_l1']),
        'seed': int(config['seed']),
        'verbosity': 0,
        'tree_method': 'hist',
    }
    if use_gpu:
        params['device'] = 'cuda'
    else:
        params['n_jobs'] = int(config['n_jobs'])
    return params
