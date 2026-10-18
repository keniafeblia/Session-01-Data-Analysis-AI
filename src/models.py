"""
models.py
Tujuan : membangun pipeline model. Semua langkah (scaling, encoding, SMOTE,
         model) dimasukkan ke SATU pipeline supaya hanya "belajar" dari data
         latih pada tiap lipatan cross-validation -> mencegah data leakage.

Kombinasi eksperimen: 3 model x 3 strategi ketidakseimbangan kelas + 1 pembanding.
  Model   : logistic regression (baseline), random forest, XGBoost
  Strategi: none | class_weight | smote
  Pembanding: "dummy" yang selalu menebak kelas mayoritas (jebakan akurasi).
"""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from config import CATEGORICAL_COLS, RANDOM_STATE

MODELS = ["logreg", "rf", "xgb"]
STRATEGIES = ["none", "class_weight", "smote"]


def build_preprocessor(columns):
    """Skala fitur numerik, one-hot untuk kategori (SEX, EDUCATION, MARRIAGE)."""
    cat = [c for c in CATEGORICAL_COLS if c in columns]
    num = [c for c in columns if c not in cat]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ]
    )


def make_estimator(model, strategy, pos_weight, quick=False):
    cw = "balanced" if strategy == "class_weight" else None
    if model == "logreg":
        return LogisticRegression(max_iter=1000, class_weight=cw, random_state=RANDOM_STATE)
    if model == "rf":
        return RandomForestClassifier(
            n_estimators=100 if quick else 300,
            min_samples_leaf=5,
            class_weight=cw,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if model == "xgb":
        return XGBClassifier(
            n_estimators=100 if quick else 300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            # untuk strategi class_weight: bobot kelas positif = rasio negatif/positif
            scale_pos_weight=pos_weight if strategy == "class_weight" else 1.0,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    raise ValueError(model)


def build_pipeline(columns, model, strategy, pos_weight, quick=False):
    steps = [("prep", build_preprocessor(columns))]
    if strategy == "smote":
        # SMOTE berada DI DALAM pipeline: hanya diterapkan pada data latih tiap fold.
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", make_estimator(model, strategy, pos_weight, quick)))
    return Pipeline(steps)


def get_configs(columns, pos_weight, quick=False):
    """Daftar (nama, model, strategi, pipeline) yang akan dibandingkan."""
    configs = [
        (
            "dummy_majority",
            "dummy",
            "none",
            Pipeline(
                [
                    ("prep", build_preprocessor(columns)),
                    ("model", DummyClassifier(strategy="prior")),
                ]
            ),
        )
    ]
    for m in MODELS:
        for s in STRATEGIES:
            configs.append((f"{m}__{s}", m, s, build_pipeline(columns, m, s, pos_weight, quick)))
    return configs
