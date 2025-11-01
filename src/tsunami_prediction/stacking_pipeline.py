from __future__ import annotations
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

def build_single_pipeline(preprocessor, estimator, use_smote: bool):
    steps = [("pre", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(k_neighbors=5, random_state=42)))
    steps.append(("clf", estimator))
    return Pipeline(steps)

def build_stacking_pipeline(preprocessor, estimators_dict: dict, use_smote: bool):
    base = [(k, v) for k, v in estimators_dict.items()]
    meta = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    stack = StackingClassifier(
        estimators=base,
        final_estimator=meta,
        stack_method="predict_proba",
        passthrough=False,
        cv=5,
        n_jobs=-1
    )
    steps = [("pre", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(k_neighbors=5, random_state=42)))
    steps.append(("stack", stack))
    return Pipeline(steps)