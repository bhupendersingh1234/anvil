import csv
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression

from anvil.data.items import Item

Predictor = Callable[[Item], float]
FEATURES = ["weight", "weight_unknown", "text_length"]


def random_pricer(train: list[Item]) -> Predictor:
    rng = random.Random(42)
    return lambda item: rng.randrange(1, 1000)


def constant_pricer(train: list[Item]) -> Predictor:
    average = sum(item.price for item in train) / len(train)
    return lambda item: average


def features(item: Item) -> dict:
    return {"weight": item.weight or 0, "weight_unknown": int(not item.weight), "text_length": len(item.summary or "")}


def linear_regression_pricer(train: list[Item]) -> Predictor:
    model = LinearRegression().fit(pd.DataFrame([features(item) for item in train])[FEATURES], [item.price for item in train])
    return lambda item: model.predict(pd.DataFrame([features(item)])[FEATURES])[0]


def bag_of_words(train: list[Item], max_features: int = 2000):
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    X = vectorizer.fit_transform([item.summary for item in train])
    return vectorizer, X, np.array([item.price for item in train], dtype=float)


def text_regressor_pricer(vectorizer, model) -> Predictor:
    return lambda item: max(0.0, float(model.predict(vectorizer.transform([item.summary]))[0]))


def nlp_linear_regression_pricer(train: list[Item]) -> Predictor:
    vectorizer, X, prices = bag_of_words(train)
    return text_regressor_pricer(vectorizer, LinearRegression().fit(X, prices))


def random_forest_pricer(train: list[Item], subset: int = 15_000) -> Predictor:
    vectorizer, X, prices = bag_of_words(train)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=4).fit(X[:subset], prices[:subset])
    return text_regressor_pricer(vectorizer, model)


def xgboost_pricer(train: list[Item]) -> Predictor:
    import xgboost as xgb

    vectorizer, X, prices = bag_of_words(train)
    model = xgb.XGBRegressor(n_estimators=1000, random_state=42, n_jobs=4, learning_rate=0.1).fit(X, prices)
    return text_regressor_pricer(vectorizer, model)


def write_human_input(test: list[Item], path: Path, size: int = 100) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([item.summary, 0] for item in test[:size])


def human_pricer(path: Path) -> Predictor:
    with path.open(encoding="utf-8") as f:
        guesses = {row[0]: float(row[1]) for row in csv.reader(f)}
    return lambda item: guesses[item.summary]


BASELINES = {
    "random": random_pricer,
    "constant": constant_pricer,
    "linear": linear_regression_pricer,
    "nlp-linear": nlp_linear_regression_pricer,
    "random-forest": random_forest_pricer,
    "xgboost": xgboost_pricer,
}
