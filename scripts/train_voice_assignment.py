#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from a2s.stage1_amt.voice_assignment_dataset import load_labeled_voice_examples
from a2s.stage1_amt.voice_assignment_model import ID_TO_VOICE, VoiceAssignmentModel
from a2s.stage1_amt.voice_assignment_features import FEATURE_NAMES
from a2s.utils.config import deep_get, load_config
from a2s.utils.json_io import save_json
from a2s.utils.logging import configure_logging


def _path(project: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project / path


def _collect(pred_dir: Path, oracle_dir: Path, onset_tol: float, offset_tol: float, onset_group_sec: float, limit: int | None = None):
    x, y, samples = [], [], []
    pred_paths = sorted(pred_dir.glob("*.json"))
    pred_paths = [path for path in pred_paths if path.name != "stage1_infer_report.json" and path.name.startswith(("train_", "val_", "test_"))]
    if limit:
        pred_paths = pred_paths[:limit]
    missing_oracles = []
    for pred_path in pred_paths:
        oracle_path = oracle_dir / pred_path.name
        if not oracle_path.exists():
            missing_oracles.append(pred_path.name)
            continue
        rows, labels, stats = load_labeled_voice_examples(pred_path, oracle_path, onset_tol, offset_tol, onset_group_sec)
        x.extend(rows)
        y.extend(labels)
        samples.append(stats)
    return x, y, samples, {
        "pred_dir": str(pred_dir),
        "oracle_dir": str(oracle_dir),
        "num_pred_files": len(pred_paths),
        "num_missing_oracles": len(missing_oracles),
        "missing_oracles_preview": missing_oracles[:20],
    }


def _build_estimator(config: dict):
    model_type = str(deep_get(config, "model.type", "hist_gradient_boosting"))
    seed = int(deep_get(config, "model.random_seed", 42))
    if model_type == "hist_gradient_boosting":
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required: pip/conda install scikit-learn") from exc
        return model_type, HistGradientBoostingClassifier(
            max_iter=int(deep_get(config, "model.max_iter", 250)),
            learning_rate=float(deep_get(config, "model.learning_rate", 0.08)),
            max_leaf_nodes=int(deep_get(config, "model.max_leaf_nodes", 31)),
            l2_regularization=float(deep_get(config, "model.l2_regularization", 0.0)),
            random_state=seed,
        )
    if model_type == "random_forest":
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required: pip/conda install scikit-learn") from exc
        return model_type, RandomForestClassifier(
            n_estimators=int(deep_get(config, "model.n_estimators", 300)),
            max_depth=deep_get(config, "model.max_depth", None),
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=int(deep_get(config, "model.n_jobs", -1)),
        )
    raise ValueError(f"unsupported voice assignment model.type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_voice_assignment.yaml")
    parser.add_argument("--train_pred_dir")
    parser.add_argument("--oracle_train_dir")
    parser.add_argument("--valid_pred_dir")
    parser.add_argument("--oracle_valid_dir")
    parser.add_argument("--out_model")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    reports_dir = _path(project, deep_get(config, "output.reports_dir", "outputs/voice_assignment"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(reports_dir / "train_voice_assignment.log")
    onset_tol = float(deep_get(config, "matching.onset_tolerance_sec", 0.05))
    offset_tol = float(deep_get(config, "matching.offset_tolerance_sec", 0.10))
    onset_group_sec = float(deep_get(config, "features.onset_group_sec", 0.04))

    train_pred_dir = _path(project, args.train_pred_dir or deep_get(config, "data.train_pred_dir", ""))
    oracle_train_dir = _path(project, args.oracle_train_dir or deep_get(config, "data.oracle_train_dir", ""))
    valid_pred_dir = _path(project, args.valid_pred_dir or deep_get(config, "data.valid_pred_dir", ""))
    oracle_valid_dir = _path(project, args.oracle_valid_dir or deep_get(config, "data.oracle_valid_dir", ""))
    logger.info("loading training examples from %s", train_pred_dir)
    x_train, y_train, train_samples, train_collect = _collect(train_pred_dir, oracle_train_dir, onset_tol, offset_tol, onset_group_sec, args.limit)
    if not x_train:
        save_json(reports_dir / "train_voice_assignment_failed_collect.json", train_collect)
        raise SystemExit(
            "no labeled training examples; run Stage 1 inference on the matching split first. "
            f"pred_dir={train_pred_dir} oracle_dir={oracle_train_dir} "
            f"missing_oracles_preview={train_collect['missing_oracles_preview']}"
        )
    logger.info("training examples=%d samples=%d labels=%s", len(x_train), len(train_samples), dict(Counter(y_train)))

    model_type, estimator = _build_estimator(config)
    estimator.fit(x_train, y_train)
    wrapped = VoiceAssignmentModel(
        estimator=estimator,
        feature_names=list(FEATURE_NAMES),
        onset_group_sec=onset_group_sec,
        constrained_onsets=bool(deep_get(config, "decode.constrained_onsets", True)),
        model_type=model_type,
    )
    out_model = _path(project, args.out_model or deep_get(config, "model.out_path", "outputs/voice_assignment/model.pkl"))
    wrapped.save(out_model)

    report = {
        "model_path": str(out_model),
        "model_type": model_type,
        "num_train_examples": len(x_train),
        "num_train_samples": len(train_samples),
        "train_collect": train_collect,
        "train_label_counts": {ID_TO_VOICE[key]: value for key, value in Counter(y_train).items()},
        "feature_names": list(FEATURE_NAMES),
    }
    if valid_pred_dir.exists() and oracle_valid_dir.exists():
        x_valid, y_valid, valid_samples, valid_collect = _collect(valid_pred_dir, oracle_valid_dir, onset_tol, offset_tol, onset_group_sec, args.limit)
        if x_valid:
            pred = estimator.predict(x_valid)
            correct = sum(int(a == b) for a, b in zip(pred, y_valid))
            report.update({
                "num_valid_examples": len(x_valid),
                "num_valid_samples": len(valid_samples),
                "valid_collect": valid_collect,
                "valid_oracle_matched_voice_accuracy": correct / len(y_valid),
                "valid_label_counts": {ID_TO_VOICE[key]: value for key, value in Counter(y_valid).items()},
            })
    save_json(reports_dir / "train_voice_assignment_report.json", report)
    logger.info("saved voice assignment model to %s", out_model)


if __name__ == "__main__":
    main()
