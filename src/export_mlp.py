"""Export the current live MLP and verify the trainer's held-out split."""
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from mlp_numpy import predict

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "models"
    model = pickle.loads((folder / "classifier.pkl").read_bytes())
    if not (isinstance(model, MLPClassifier) and len(model.coefs_) == 2
            and model.activation == "relu" and model.out_activation_ == "softmax"):
        raise ValueError("Expected the single-hidden-layer ReLU and softmax classifier.")
    data = pd.read_csv(ROOT / "data" / "landmarks.csv", keep_default_na=False)
    columns = [f"{axis}{i}" for i in range(21) for axis in "xyz"]
    x = data[columns].to_numpy(np.float32)
    y = data["label"].str.removeprefix("\ufeff")
    train, test, _, _ = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)
    if model.t_ != len(train) * model.n_iter_:
        raise ValueError("Dataset size differs from the fitted model's training split.")
    expected = model.predict(test)
    live = ort.InferenceSession(str(folder / "classifier.onnx"), providers=["CPUExecutionProvider"])
    labels, probabilities = live.run(None, {live.get_inputs()[0].name: test})
    np.testing.assert_array_equal(labels.ravel(), expected)
    np.testing.assert_allclose(probabilities, model.predict_proba(test), atol=1e-5, rtol=1e-5)
    with TemporaryDirectory(dir=folder) as tmp:
        archive = Path(tmp) / "mlp.npz"
        np.savez_compressed(archive, classes=model.classes_.astype(str),
                            activation=model.activation, output_activation=model.out_activation_,
                            weights_0=model.coefs_[0], weights_1=model.coefs_[1],
                            bias_0=model.intercepts_[0], bias_1=model.intercepts_[1])
        with np.load(archive, allow_pickle=False) as saved:
            np.testing.assert_array_equal(predict(saved, test), expected)
            for row, label in zip(test, expected):
                np.testing.assert_array_equal(predict(saved, row), [label])
        archive.replace(folder / archive.name)
    print(f"All {len(test)} test labels matched in batch and individually.")


if __name__ == "__main__":
    main()
