"""Summarize Section 5.2 metrics and paired AP differences."""
import argparse
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t

from tscb.sota_delay import METHODS
from tscb.utils import METRICS

TABLE_METHODS = ('full', 'static_tuned', 'backbone', *METHODS)
WINDOWS = range(5)


def load_runs(root, seed):
    folders = sorted(root.glob('seed-*')) if seed is None else [root / f'seed-{seed}']
    if not folders:
        raise ValueError(f'No seed directories in {root}')
    runs = []
    for folder in folders:
        run = []
        for fold in WINDOWS:
            path = folder / f'fold-{fold}.json'
            record = json.loads(path.read_text(encoding='utf-8'))
            if record['fold'] != fold or folder.name != f"seed-{record['seed']}":
                raise ValueError(f'Seed or window mismatch in {path}')
            run.append(record)
        runs.append(run)
    return runs


def average_windows(runs, setting, arm, metric):
    return [mean(run[i][setting][arm][metric] for run in runs) for i in WINDOWS]


def paired_ap(runs, setting, arm):
    differences = [mean(run[i][setting]['full']['ap'] - run[i][setting][arm]['ap']
                        for run in runs) for i in WINDOWS]
    n = len(differences)
    delta = mean(differences)
    se = stdev(differences) / sqrt(n)
    half = float(t.ppf(0.975, n - 1)) * se
    p = float(2 * t.sf(abs(delta / se), n - 1)) if se else None
    return delta, delta - half, delta + half, p


def report(runs):
    print(f'Section 5.2: {len(runs)} seed(s), {len(WINDOWS)} windows\n')
    print('14-day delay\n\n| Method | AP | Brier | AUROC | P@2% |')
    print('| --- | ---: | ---: | ---: | ---: |')
    for arm in TABLE_METHODS:
        values = [mean(average_windows(runs, '14_day', arm, m)) for m in METRICS]
        print(f'| {arm} | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')

    print('\nPer-window recovery (AP)\n')
    print('| Window | Matured positives | Gated | full | static_tuned | wait_tuned |')
    print('| ---: | ---: | --- | ---: | ---: | ---: |')
    curves = [average_windows(runs, '14_day', arm, 'ap')
              for arm in ('full', 'static_tuned', 'wait_tuned')]
    for i in WINDOWS:
        counts = {(run[i]['matured_positives'], run[i]['gated']) for run in runs}
        if len(counts) != 1:
            raise ValueError(f'Maturity differs across seeds in window {i + 1}')
        (positives, gated), = counts
        values = ' | '.join(f'{curve[i]:.4f}' for curve in curves)
        print(f"| {i + 1} | {positives} | {'yes' if gated else 'no'} | {values} |")

    print('\nNo delay (mean AP)\n')
    for arm in ('full', 'static_tuned'):
        print(f"{arm}: {mean(average_windows(runs, 'no_delay', arm, 'ap')):.4f}")
    print('\nPaired AP differences (full minus reference)\n')
    print('| Setting | Reference | Delta AP | 95% CI | p |')
    print('| --- | --- | ---: | ---: | ---: |')
    for setting, refs in (('14_day', (*TABLE_METHODS[1:], 'wait_tuned')),
                          ('no_delay', ('static_tuned',))):
        for arm in refs:
            delta, low, high, p = paired_ap(runs, setting, arm)
            p_value = f'{p:.4g}' if p is not None else 'n/a'
            print(f'| {setting} | {arm} | {delta:+.4f} | [{low:+.4f}, {high:+.4f}] | {p_value} |')


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('results', type=Path, nargs='?', default=Path('results'))
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    try:
        report(load_runs(args.results, args.seed))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Cannot summarize results: {exc}\n')


if __name__ == '__main__':
    main()
