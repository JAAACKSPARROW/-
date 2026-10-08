#!/usr/bin/env python3
"""Pair two evaluation CSVs by experiment/model/seed without retraining or overwriting."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

KEYS = ['experiment', 'model', 'seed']
DEFAULT_METRICS = ['accuracy', 'balanced_accuracy', 'precision', 'recall',
                   'specificity', 'f1', 'mcc', 'roc_auc', 'pr_auc']


def load(path, metrics):
    rows = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ValueError(f'{path}: duplicate CSV column names')
        missing = set(KEYS + metrics) - set(headers)
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        for line, row in enumerate(reader, 2):
            key = tuple(row.get(k) for k in KEYS)
            if any(value is None or not value.strip() for value in key):
                raise ValueError(f'{path}:{line}: missing identity key')
            if key in rows:
                raise ValueError(f'{path}:{line}: duplicate key {key}')
            values = {}
            for metric in metrics:
                try:
                    value = float(row[metric])
                except (ValueError, TypeError):
                    raise ValueError(f'{path}:{line}: invalid numeric metric {metric}') from None
                if not math.isfinite(value):
                    raise ValueError(f'{path}:{line}: non-finite metric {metric}')
                values[metric] = value
            rows[key] = values
    if not rows:
        raise ValueError(f'{path}: no evaluation rows')
    return rows


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(first, second, output, metrics, tolerance):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output already exists; use a new comparison directory: {output}')
    if not metrics or len(metrics) != len(set(metrics)) or set(metrics) & set(KEYS):
        raise ValueError('Choose nonempty, unique metric columns distinct from identity keys')
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Tolerance must be finite and nonnegative')
    one, two = load(first, metrics), load(second, metrics)
    if one.keys() != two.keys():
        only_first, only_second = set(one) - set(two), set(two) - set(one)
        raise ValueError(f'Unpaired evaluations: only run1={len(only_first)}, only run2={len(only_second)}; '
                         f'examples={sorted(only_first)[:2]} / {sorted(only_second)[:2]}')
    paired, groups = [], defaultdict(list)
    for key in sorted(one):
        row = dict(zip(KEYS, key))
        for metric in metrics:
            x, y = one[key][metric], two[key][metric]
            delta = y - x
            if not math.isfinite(delta):
                raise ValueError(f'Numeric overflow for {key}/{metric}')
            row.update({f'{metric}_run1': x, f'{metric}_run2': y, f'{metric}_delta': delta})
            groups[(key[0], key[1], metric)].append((x, y, delta))
        paired.append(row)
    summaries = []
    changed = set()
    for (experiment, model, metric), values in sorted(groups.items()):
        x, y, delta = map(list, zip(*values))
        count = sum(abs(d) > tolerance for d in delta)
        if count:
            changed.add((experiment, model))
        summaries.append(dict(experiment=experiment, model=model, metric=metric, n=len(values),
                              run1_mean=statistics.mean(x), run1_std=statistics.stdev(x) if len(x) > 1 else '',
                              run2_mean=statistics.mean(y), run2_std=statistics.stdev(y) if len(y) > 1 else '',
                              mean_delta=statistics.mean(delta), max_abs_seed_delta=max(map(abs, delta)),
                              changed_seeds=count, equal_within_tolerance=count == 0))
    metadata = dict(paired_rows=len(paired), metrics=metrics, keys=KEYS,
                    absolute_tolerance=tolerance, relative_tolerance=0, std_ddof=1,
                    run1_sha256=hashlib.sha256(first.read_bytes()).hexdigest(),
                    run2_sha256=hashlib.sha256(second.read_bytes()).hexdigest(),
                    changed_experiment_models=[dict(experiment=e, model=m) for e, m in sorted(changed)],
                    limitation='Metric comparison only; verify protocol, sample IDs, environments and model provenance separately.')
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'paired_metrics.csv', paired)
    write_csv(output / 'metric_summary.csv', summaries)
    (output / 'comparison.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run1', required=True, type=Path)
    parser.add_argument('--run2', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS)
    parser.add_argument('--tolerance', type=float, default=1e-12)
    args = parser.parse_args()
    try:
        result = compare(args.run1, args.run2, args.output, args.metrics, args.tolerance)
    except (ValueError, OSError) as error:
        parser.exit(2, f'Comparison stopped: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
