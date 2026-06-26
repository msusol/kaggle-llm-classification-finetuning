#!/usr/bin/env python3
"""Build TF-IDF + length features for the LLM classification baseline."""
import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from log_utils import setup_logging

DATA = Path(__file__).parent.parent / "data"
FEATS = DATA / "features"


def load_data(train_path: Path, test_path: Path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def label_encode(df: pd.DataFrame) -> np.ndarray:
    # 0=model_a, 1=model_b, 2=tie
    return (
        df[["winner_model_a", "winner_model_b", "winner_tie"]]
        .values.argmax(axis=1)
        .astype(np.int32)
    )


def make_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["prompt"].fillna("") + " [SEP] "
        + df["response_a"].fillna("") + " [SEP] "
        + df["response_b"].fillna("")
    )


def length_features(df: pd.DataFrame) -> np.ndarray:
    a = df["response_a"].fillna("").str.len().astype(float)
    b = df["response_b"].fillna("").str.len().astype(float)
    p = df["prompt"].fillna("").str.len().astype(float)
    return np.column_stack([
        a, b, p,
        a - b,                          # len_diff
        a / (b + 1),                    # len_ratio
        (a == 0).astype(float),         # empty_a
        (b == 0).astype(float),         # empty_b
    ])


def build(train_csv: Path, test_csv: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    train, test = load_data(train_csv, test_csv)

    y = label_encode(train)
    np.save(out_dir / "y_train.npy", y)
    np.save(out_dir / "train_ids.npy", train["id"].values)
    np.save(out_dir / "test_ids.npy", test["id"].values)

    train_text = make_text(train)
    test_text = make_text(test)

    logging.info("Fitting TF-IDF word n-grams …")
    word_tfidf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_features=50_000,
        sublinear_tf=True, min_df=5,
    )
    X_word_train = word_tfidf.fit_transform(train_text)
    X_word_test = word_tfidf.transform(test_text)

    len_train = length_features(train)
    len_test = length_features(test)
    from scipy.sparse import csr_matrix
    X_train = hstack([X_word_train, csr_matrix(len_train)])
    X_test = hstack([X_word_test, csr_matrix(len_test)])

    save_npz(out_dir / "X_train.npz", X_train)
    save_npz(out_dir / "X_test.npz", X_test)

    with open(out_dir / "tfidf_word.pkl", "wb") as f:
        pickle.dump(word_tfidf, f)

    logging.info("X_train: %s  X_test: %s", X_train.shape, X_test.shape)
    logging.info("y_train class counts: %s", np.bincount(y))
    logging.info("Features saved to %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(DATA / "train.csv"))
    parser.add_argument("--test", default=str(DATA / "test.csv"))
    parser.add_argument("--out", default=str(FEATS))
    args = parser.parse_args()
    setup_logging(__file__)
    build(Path(args.train), Path(args.test), Path(args.out))
