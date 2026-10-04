from pathlib import Path
import json

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from preprocessing import preprocess_subject
from csp_features import CSPFeatureExtractor


SUBJECTS = list(range(1, 11))

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    all_X = []
    all_y = []
    all_groups = []

    print("=" * 60)
    print("NEUROTWIN v2.0 - CSP + LDA")
    print("=" * 60)

    for subject in SUBJECTS:
        print(f"\nProcessing subject {subject}...")

        epochs = preprocess_subject(subject)

        y = epochs.events[:, -1]
        unique = np.unique(y)

        if len(unique) != 2:
            raise RuntimeError(
                f"Subject {subject}: expected 2 classes, found {unique}"
            )

        # Convert arbitrary event IDs to 0/1
        mapping = {unique[0]: 0, unique[1]: 1}
        y = np.array([mapping[value] for value in y])

        X = epochs.get_data(copy=True)

        all_X.append(X)
        all_y.append(y)
        all_groups.extend([subject] * len(y))

        print(f"  Epochs: {len(y)}")
        print(f"  Channels: {X.shape[1]}")
        print(f"  Time points: {X.shape[2]}")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y)
    groups = np.asarray(all_groups)

    print("\n" + "=" * 60)
    print("DATASET")
    print("=" * 60)
    print(f"Total epochs: {len(y)}")
    print(f"Channels: {X.shape[1]}")
    print(f"Time points: {X.shape[2]}")
    print(f"Subjects: {len(np.unique(groups))}")

    cv = GroupKFold(n_splits=5)

    fold_results = []
    all_true = []
    all_pred = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):
        print(f"\n--- Fold {fold}/5 ---")

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        csp = CSPFeatureExtractor(n_components=6)

        X_train_csp = csp.fit_transform(
            _EpochWrapper(X_train),
            y_train,
        )

        X_test_csp = csp.transform(
            _EpochWrapper(X_test),
        )

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LinearDiscriminantAnalysis()),
            ]
        )

        model.fit(X_train_csp, y_train)

        pred = model.predict(X_test_csp)

        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(
            y_test, pred, zero_division=0
        )
        recall = recall_score(
            y_test, pred, zero_division=0
        )
        f1 = f1_score(
            y_test, pred, zero_division=0
        )

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1       : {f1:.4f}")

        fold_results.append(
            {
                "fold": fold,
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

        all_true.extend(y_test.tolist())
        all_pred.extend(pred.tolist())

    cm = confusion_matrix(all_true, all_pred)

    summary = {
        "model": "CSP + LDA",
        "subjects": SUBJECTS,
        "total_epochs": int(len(y)),
        "folds": 5,
        "fold_results": fold_results,
        "mean_accuracy": float(
            np.mean([x["accuracy"] for x in fold_results])
        ),
        "mean_precision": float(
            np.mean([x["precision"] for x in fold_results])
        ),
        "mean_recall": float(
            np.mean([x["recall"] for x in fold_results])
        ),
        "mean_f1": float(
            np.mean([x["f1"] for x in fold_results])
        ),
        "confusion_matrix": cm.tolist(),
    }

    output = RESULTS_DIR / "v2_metrics.json"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Mean Accuracy: {summary['mean_accuracy']:.4f}")
    print(f"Mean F1:       {summary['mean_f1']:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    print(f"\nSaved: {output}")


class _EpochWrapper:
    """
    Small adapter so CSP can receive an object
    exposing get_data().
    """

    def __init__(self, data):
        self.data = data

    def get_data(self, copy=True):
        return self.data.copy() if copy else self.data


if __name__ == "__main__":
    main()