#!/usr/bin/env python3
"""5-fold LightGBM multiclass baseline. Reads features from data/features/."""
import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from log_utils import setup_logging

DATA = Path(__file__).parent.parent / "data"
FEATS = DATA / "features"

LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": -1,
    "verbose": -1,
    "seed": 42,
}


def run(feat_dir: Path, n_folds: int, n_rounds: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = load_npz(feat_dir / "X_train.npz")
    X_test = load_npz(feat_dir / "X_test.npz")
    y = np.load(feat_dir / "y_train.npy")
    test_ids = np.load(feat_dir / "test_ids.npy", allow_pickle=True)

    oof_preds = np.zeros((len(y), 3))
    test_preds = np.zeros((len(test_ids), 3))
    fold_scores = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_bar = tqdm(skf.split(X_train, y), total=n_folds, desc="Folds", unit="fold")
    for fold, (tr_idx, val_idx) in enumerate(fold_bar):
        logging.info("── Fold %d/%d ──", fold + 1, n_folds)
        dtrain = lgb.Dataset(X_train[tr_idx], label=y[tr_idx])
        dval = lgb.Dataset(X_train[val_idx], label=y[val_idx], reference=dtrain)

        round_bar = tqdm(total=n_rounds, desc=f"  Fold {fold+1} rounds", unit="round", leave=False)

        def _tqdm_callback(env):
            round_bar.update(1)
            if (env.iteration + 1) % 25 == 0 or env.iteration + 1 == env.end_iteration:
                val_loss = env.evaluation_result_list[0][2] if env.evaluation_result_list else float("nan")
                logging.info("Fold %d | round %d | val log_loss: %.4f", fold + 1, env.iteration + 1, val_loss)
            if env.iteration + 1 == env.end_iteration:
                round_bar.close()

        _tqdm_callback.order = 10

        model = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=n_rounds,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
                _tqdm_callback,
            ],
        )

        oof_preds[val_idx] = model.predict(X_train[val_idx])
        test_preds += model.predict(X_test) / n_folds

        fold_ll = log_loss(y[val_idx], oof_preds[val_idx])
        fold_scores.append(fold_ll)
        fold_bar.set_postfix({"log_loss": f"{fold_ll:.4f}"})
        logging.info("Fold %d log loss: %.4f  (best iter: %d)", fold + 1, fold_ll, model.best_iteration)

    oof_ll = log_loss(y, oof_preds)
    logging.info("OOF log loss: %.4f  (folds: %s)", oof_ll, [f"{s:.4f}" for s in fold_scores])

    # Write submission
    sub = pd.DataFrame({
        "id": test_ids,
        "winner_model_a": test_preds[:, 0],
        "winner_model_b": test_preds[:, 1],
        "winner_tie":     test_preds[:, 2],
    })
    sub_path = out_dir / "submission.csv"
    sub.to_csv(sub_path, index=False)
    logging.info("Submission written to %s", sub_path)

    # Save OOF for leaderboard tracking
    np.save(out_dir / "oof_preds.npy", oof_preds)
    results = {"oof_log_loss": round(oof_ll, 4), "fold_scores": [round(s, 4) for s in fold_scores]}
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logging.info("Results: %s", json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat-dir", default=str(FEATS))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-rounds", type=int, default=300)
    parser.add_argument("--out", default=str(DATA.parent / "output" / "v0.1-baseline"))
    args = parser.parse_args()
    setup_logging(__file__)
    run(Path(args.feat_dir), args.n_folds, args.n_rounds, Path(args.out))
