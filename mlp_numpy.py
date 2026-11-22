"""NumPy inference for the ASL MLP."""
import numpy as np

if __package__:
    from .normalization import normalize
else:
    from normalization import normalize


def predict_proba(model, features):
    """Return class probabilities for raw (21, 3) landmarks or normalized (63,) / (N, 63) features."""
    x = np.asarray(features, dtype=np.float32)
    if x.shape == (21, 3):
        x = normalize(x)
    x = np.atleast_2d(x)
    if x.ndim != 2 or x.shape[1] != 63:
        raise ValueError("Expected raw (21, 3) landmarks or normalized 63-feature rows.")
    hidden = np.maximum(x @ model["weights_0"] + model["bias_0"], 0)
    logits = hidden @ model["weights_1"] + model["bias_1"]
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def predict(model, features):
    return model["classes"][predict_proba(model, features).argmax(axis=1)]
