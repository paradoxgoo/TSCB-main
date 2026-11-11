"""Compare configuration recovery with frozen recipes on five causal windows."""
import json
from pathlib import Path

from . import tree_framework as tree
from .sota_delay import METHODS, fit_external
from .utils import build_features, evaluate, fit_scale, load_dataset, rolling_time_folds


DELAY_SECONDS = 14 * 86400


def run(data_dir, output, seed, cpu):
    config = json.loads((Path(__file__).parent / 'params' / 'ieee.json').read_text(encoding='utf-8'))
    config['_use_gpu'] = False
    if not cpu:
        import torch

        config['_use_gpu'] = torch.cuda.is_available()
    frame, labels, times = load_dataset(data_dir)
    time_values = times.to_numpy()
    labels = labels.to_numpy()
    output.mkdir(parents=True, exist_ok=True)
    delayed_recipe = wait_recipe = clean_recipe = None

    for fold in rolling_time_folds(times):
        train, valid, test = fold['train'], fold['validation'], fold['test']
        values = fit_scale(build_features(frame, times, train), train)
        cutoff = fold['validation_end'] - DELAY_SECONDS
        mature_train = train[time_values[train] <= cutoff]
        mature_valid = valid[time_values[valid] <= cutoff]
        matured_positives = int(labels[mature_valid].sum())
        gated = matured_positives < config['min_calibration_positives']
        test_x = values[test]
        delayed = (values[mature_train], labels[mature_train],
                   values[mature_valid], labels[mature_valid])
        full_recipe = tree.select_recipe(*delayed, config, seed)
        if delayed_recipe is None:
            delayed_recipe = full_recipe
        if wait_recipe is None and not gated:
            wait_recipe = full_recipe

        delayed_scores = {}
        for arm, recipe in (
            ('full', full_recipe),
            ('static_tuned', delayed_recipe),
            ('backbone', tree.backbone_recipe(*delayed, config, seed)),
            ('wait_tuned', wait_recipe or tree.default_recipe(delayed[1], config)),
        ):
            scores = tree.fit_static(*delayed, test_x, config, seed, recipe)
            delayed_scores[arm] = evaluate(labels[test], scores)
        for method in METHODS:
            scores = fit_external(method, *delayed, test_x, config, seed)
            delayed_scores[method] = evaluate(labels[test], scores)

        clean = (values[train], labels[train], values[valid], labels[valid])
        full_recipe = tree.select_recipe(*clean, config, seed)
        if clean_recipe is None:
            clean_recipe = full_recipe
        clean_scores = {}
        for arm, recipe in (('full', full_recipe), ('static_tuned', clean_recipe)):
            scores = tree.fit_static(*clean, test_x, config, seed, recipe)
            clean_scores[arm] = evaluate(labels[test], scores)

        del values, delayed, clean, test_x
        record = {
            'seed': seed,
            'fold': fold['fold'],
            'matured_positives': matured_positives,
            'gated': gated,
            '14_day': delayed_scores,
            'no_delay': clean_scores,
        }
        path = output / f"fold-{fold['fold']}.json"
        path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(f'Wrote {path}', flush=True)
