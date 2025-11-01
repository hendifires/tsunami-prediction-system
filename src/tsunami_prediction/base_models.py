from __future__ import annotations
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

def get_base_estimators(random_state=42):
    return {
        "dt": DecisionTreeClassifier(max_depth=None, random_state=random_state),
        "rf": RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced",
                                     random_state=random_state),
        "svm": SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced",
                   random_state=random_state),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "nb": GaussianNB(),
        "ann": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=random_state),
        "gb": GradientBoostingClassifier(random_state=random_state),
    }