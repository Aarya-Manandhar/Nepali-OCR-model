# eval_ours.py -- CER of best.pt on a labeled folder of line images
# usage: python eval_ours.py val_dataset
import argparse
from pathlib import Path
from predict import read_line

def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

parser = argparse.ArgumentParser(description="Calculate character error rate for a labeled line-image folder")
parser.add_argument("folder", help="folder containing labels.txt and referenced images")
args = parser.parse_args()
folder = Path(args.folder)
rows = [l.rstrip("\n").split("\t", 1) for l in (folder / "labels.txt").open(encoding="utf-8") if "\t" in l]
errs = tot = 0
for path, truth in rows:
    pred = read_line(str(folder / path))
    errs += lev(pred, truth); tot += len(truth)
print("best.pt CER on", len(rows), "lines in", folder, ":", round(errs / tot, 4))
