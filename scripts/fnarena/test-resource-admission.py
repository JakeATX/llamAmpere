"""Reject absent, stale and malformed responsiveness evidence before launch."""
import json
import tempfile
from pathlib import Path

from resource_admission import calm_monitor


def main():
    now = 1000.0
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'latest.json'
        assert not calm_monitor(path, now)
        cases = [
            ({'timestamp': 999, 'sustained_alert': False, 'reasons': []}, True),
            ({'timestamp': 970, 'sustained_alert': False, 'reasons': []}, True),
            ({'timestamp': 969, 'sustained_alert': False, 'reasons': []}, False),
            ({'timestamp': 1001, 'sustained_alert': False, 'reasons': []}, False),
            ({'timestamp': 999, 'sustained_alert': True, 'reasons': []}, False),
            ({'timestamp': 999, 'sustained_alert': False, 'reasons': ['memory pressure']}, False),
            ({'timestamp': 999, 'sustained_alert': 0, 'reasons': []}, False),
            ({'timestamp': 999, 'reasons': []}, False),
            ({'timestamp': 999, 'sustained_alert': False}, False),
            ({'timestamp': '999', 'sustained_alert': False, 'reasons': []}, False),
            ({'timestamp': True, 'sustained_alert': False, 'reasons': []}, False),
            ({'timestamp': float('nan'), 'sustained_alert': False, 'reasons': []}, False),
            ({'timestamp': float('inf'), 'sustained_alert': False, 'reasons': []}, False),
            ({'timestamp': 10**1000, 'sustained_alert': False, 'reasons': []}, False),
            ([], False),
            ({}, False),
        ]
        for record, expected in cases:
            path.write_text(json.dumps(record))
            assert calm_monitor(path, now) is expected, record
        for raw in ['{', 'x'*65537]:
            path.write_text(raw)
            assert not calm_monitor(path, now)
        path.write_text(json.dumps(cases[0][0]))
        assert not calm_monitor(path, float('nan'))
        assert not calm_monitor(path, now, max_age=-1)
    print('PASS resource admission: fresh explicit calm only; 21 cases')


if __name__ == '__main__':
    main()
