import jsonlines
from pathlib import Path
paths = [str(x) for x in Path('').glob('**/*.jsonl')]

fact_count = 0
for path in paths:
    with jsonlines.open(path) as f:
        for line in f.iter():
            if "sub_label" in line and "obj_label" in line:
                fact_count += 1
print(fact_count)