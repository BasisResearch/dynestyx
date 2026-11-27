import json
from pathlib import Path
import sys

cases = json.loads(Path(sys.argv[1]).read_text())['cases']
if len(sys.argv) == 2:
    lengths = set()
    for label, rows in cases.items():
        group, length = label.split('/')
        base = rows['manual' if group=='sequential' else 'native']['median_ms']
        if any(r['median_ms'] > 1.10*base for r in rows.values()):
            lengths.add(int(length))
    print(*sorted(lengths))
else:
    candidate = json.loads(Path(sys.argv[2]).read_text())['cases']
    def median(rows):
        return next(iter(rows.values()))['median_ms']
    print(*(label for label in cases if median(candidate[label]) > 1.10*median(cases[label])))
