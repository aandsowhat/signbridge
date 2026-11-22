"""Train and export a static fingerspelling classifier."""
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from skl2onnx import to_onnx

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data = pd.read_csv(ROOT / "data" / "landmarks.csv", keep_default_na=False)
    columns = [f"{axis}{i}" for i in range(21) for axis in "xyz"]
    x = data[columns].to_numpy(dtype=np.float32)
    y = data["label"].str.removeprefix("\ufeff").to_numpy(dtype=str)
    if not np.isfinite(x).all():
        raise ValueError("The CSV contains non-finite landmark coordinates.")
    if len(np.unique(y)) < 2:
        raise ValueError("Record at least two different letters before training.")

    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42)
    fit_x, val_x, fit_y, val_y = train_test_split(
        train_x, train_y, test_size=0.2, stratify=train_y, random_state=42)
    candidates = [RandomForestClassifier(n_estimators=100, random_state=42),
                  MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42)]
    scores = []
    for model in candidates:
        model.fit(fit_x, fit_y)
        score = model.score(val_x, val_y)
        scores.append(score)
        print(f"{type(model).__name__} validation accuracy: {score:.2%}", flush=True)

    winner = candidates[np.argmax(scores)].fit(train_x, train_y)
    predictions = winner.predict(test_x)
    labels = winner.classes_
    matrix = confusion_matrix(test_y, predictions, labels=labels)
    print(f"{type(winner).__name__} test accuracy: {accuracy_score(test_y, predictions):.2%}")
    for label, correct, total in zip(labels, matrix.diagonal(), matrix.sum(axis=1)):
        print(f"{label}: {correct}/{total} ({correct / total:.1%})")
    print("Confusion matrix (rows = actual, columns = predicted)")
    print(pd.DataFrame(matrix, index=labels, columns=labels).to_string())

    folder = ROOT / "models"
    folder.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=folder) as temp:
        pkl = Path(temp) / "classifier.pkl"
        onnx = Path(temp) / "classifier.onnx"
        pkl.write_bytes(pickle.dumps(winner))
        exported = to_onnx(winner, test_x[:1], options={id(winner): {"zipmap": False}},
                           target_opset=17)
        onnx.write_bytes(exported.SerializeToString())
        session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
        actual = session.run(None, {session.get_inputs()[0].name: test_x})[0]
        expected = pickle.loads(pkl.read_bytes()).predict(test_x)
        np.testing.assert_array_equal(actual.ravel(), expected)
        pkl.replace(folder / pkl.name)
        onnx.replace(folder / onnx.name)
    print(f"Saved both models. Predictions matched on {len(test_y)} test frames.")


if __name__ == "__main__":
    main()
