"""Compute exact-count competing-event incidence from a JSON counting table."""

import argparse
import json
from pathlib import Path
import sys

from .core import EventRow, EventTable, estimate


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args(argv)
    try:
        with args.input.open('rb') as handle:
            data = handle.read(1024*1024+1)
        if len(data) > 1024*1024:
            raise ValueError('input exceeds 1 MiB')
        value = json.loads(data, object_pairs_hook=_pairs)
        if not isinstance(value, dict) or not {'initial','causes','rows'} <= value.keys() or value.keys()-{'initial','causes','rows','query_times'}:
            raise ValueError('missing or unknown input fields')
        if not isinstance(value['rows'], list) or not 1 <= len(value['rows']) <= 200:
            raise ValueError('rows must be a list of length 1 to 200')
        rows = []
        for row in value['rows']:
            if not isinstance(row, dict) or set(row) != {'time','events','censored'}:
                raise ValueError('missing or unknown row fields')
            rows.append(EventRow(**row))
        result = estimate(EventTable(value['initial'], value['causes'], rows))
        report = result.to_dict()
        if 'query_times' in value:
            times = value['query_times']
            if not isinstance(times, list) or not 1 <= len(times) <= 1000:
                raise ValueError('query_times must have length 1 to 1000')
            report['queries'] = [result.at(t) for t in times]
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        print(f'competing-events: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
