"""Export a trusted sklearn ASL forest and verify its original seeded test split."""
import argparse
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from forest_numpy import predict


def export(model, path):
    trees = [estimator.tree_ for estimator in model.estimators_]
    roots = np.cumsum([0] + [tree.node_count for tree in trees[:-1]])

    def children(side):
        parts = []
        for tree, root in zip(trees, roots):
            local = getattr(tree, side)
            parts.append(np.where(local >= 0, local + root, -1))
        return np.concatenate(parts)

    values = np.concatenate([tree.value[:, 0, :] for tree in trees])
    np.savez_compressed(path, classes=model.classes_.astype(str), roots=roots,
                        n_features=model.n_features_in_, children_left=children("children_left"),
                        children_right=children("children_right"), values=values,
                        feature=np.concatenate([tree.feature for tree in trees]),
                        threshold=np.concatenate([tree.threshold for tree in trees]),
                        missing_go_to_left=np.concatenate([tree.missing_go_to_left for tree in trees]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = pickle.loads(args.pickle.read_bytes())
    if not isinstance(model, RandomForestClassifier):
        parser.error("Supply a trained RandomForestClassifier pickle.")
    data = pd.read_csv(args.csv)
    x = data[[f"{axis}{i}" for i in range(21) for axis in "xyz"]].to_numpy(np.float32)
    y = data["label"].str.removeprefix("\ufeff")
    _, test_x, _, test_y = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        path = Path(temp) / "forest.npz"
        export(model, path)
        with np.load(path, allow_pickle=False) as archive:
            actual = predict(archive, test_x)
            np.testing.assert_array_equal(actual, model.predict(test_x))
        path.replace(args.output)
    print(f"{len(actual)} matching predictions; {(actual == test_y).sum()} correct.")
