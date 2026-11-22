"""Run the exported ASL random forest with NumPy alone."""
import numpy as np


def predict(model, features):
    """Return one label per row of normalized landmark features (one row also accepted)."""
    x = np.atleast_2d(np.asarray(features, dtype=np.float32))
    if x.ndim != 2 or x.shape[1] != int(model["n_features"]):
        raise ValueError("Expected one row of landmark features per sample.")
    feature, threshold = model["feature"], model["threshold"]
    left, right = model["children_left"], model["children_right"]
    missing_left, values = model["missing_go_to_left"], model["values"]
    scores = np.zeros((len(x), values.shape[1]), dtype=np.float64)
    for root in model["roots"]:
        nodes = np.full(len(x), root)
        while (rows := np.flatnonzero(left[nodes] >= 0)).size:
            parents = nodes[rows]
            observed = x[rows, feature[parents]]
            go_left = np.where(np.isnan(observed), missing_left[parents],
                               observed <= threshold[parents])
            nodes[rows] = np.where(go_left, left[parents], right[parents])
        scores += values[nodes]
    return model["classes"][scores.argmax(axis=1)]
