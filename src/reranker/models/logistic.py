import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class LogisticReranker:
    """
    Wrapper around a logistic regression reranking model.
    """

    DEFAULT_NUM_COLS = ["cosine", "hellinger"]
    DEFAULT_CAT_COLS = ["help_center_id", "article_id"]

    def __init__(
            self,
            random_state: int = 42,
            solver: str = "saga",
            max_iter: int = 2000,
            penalty: str = "l2",
            C: float = 0.1,
            class_weight: str | dict = "balanced",
            n_jobs: int = -1,
    ):
        self.random_state = random_state
        self.hyperparams = dict(
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            penalty=penalty,
            C=C,
        )
        self.model = self._build_pipeline()
        self.is_fitted = False

    def _build_pipeline(self) -> Pipeline:
        """Build preprocessing + classifier pipeline."""
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.DEFAULT_NUM_COLS),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), self.DEFAULT_CAT_COLS),
            ]
        )

        clf = LogisticRegression(**self.hyperparams)
        return Pipeline([("preprocessor", preprocessor), ("clf", clf)])

    def fit(self, train_df: pd.DataFrame, y):
        """Fits the logistic regression reranker."""
        self.model.fit(train_df, y)
        self.is_fitted = True
        return self

    def predict_proba(self, train_df: pd.DataFrame):
        """Predicts relevance probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet.")
        return self.model.predict_proba(train_df)
