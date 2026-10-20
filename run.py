"""Regenerate the submission from the provided data.

    python run.py [--out out/submission.csv]

Reads the data bundle from ./data (or $MTAH_DATA_DIR). Deterministic: the same data gives the
same file.
"""
import argparse
import hashlib

from possession import data, pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/submission.csv")
    args = parser.parse_args()
    labels, _ = pipeline.run()
    path = data.write_submission(args.out, labels)
    with open(path, "rb") as handle:
        digest = hashlib.md5(handle.read()).hexdigest()
    print("wrote {}  md5 {}".format(path, digest))
