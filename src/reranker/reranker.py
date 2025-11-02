import logging
import os
from datetime import datetime
from typing import Dict, List
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from reranker.eval.metrics import compute_metrics
from reranker.features.build import build_candidate_features_inference
from reranker.features.build import build_training_examples, build_candidate_features
from reranker.models.logistic import LogisticReranker
from reranker.utils.embeddings import mean_pool_query


class Reranker:
    """
    Orchestrator class for training and evaluating article reranking models.
    """

    def __init__(
            self,
            model_name: str = "logistic",
            n_negatives: int = 3,
            eval_k: int = 10,
            random_state: int = 42,
    ):
        self.model_name = model_name
        self.n_negatives = n_negatives
        self.eval_k = eval_k
        self.random_state = random_state

        self._model = self._build_model()
        self._is_fitted = False

        self.logger = logging.getLogger(__name__)

    def _build_model(self):
        """Instantiate the chosen reranker model."""
        if self.model_name == "logistic":
            return LogisticReranker(random_state=self.random_state)
        raise ValueError(f"Unknown model: {self.model_name}")

    def fit(self, emb_df, arr, fb_df):
        """Main training logic: build features + fit model."""
        self.logger.info("Building training examples...")
        x, y = build_training_examples(fb_df, emb_df, arr, n_random_negatives=self.n_negatives)
        self.logger.info("Training reranker model...")
        self._model.fit(x, y)
        self._is_fitted = True
        self.logger.info("Training completed successfully.")
        return self

    def evaluate(self, fb_df, article_emb_df, article_emb_matrix, top_n_cosine: int = 30) -> Dict:
        """
        Evaluates reranker performance on feedback data.
        Simulates production: cosine pre-filter, rerank top-N, and compare both.
        Returns mean metrics (hit@k, MRR) for both cosine & model reranker.
        """
        self.logger.info("Evaluating on validation set...")
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet.")

        rows = []
        present_ids = set(article_emb_df["id"])
        emb_ids = article_emb_df["id"].to_numpy()
        hc_arr = article_emb_df["help_center_id"].to_numpy()

        for _, r in fb_df.iterrows():
            query_chunks = r["query_embeddings_parsed"]
            hc = r["help_center_id"]
            cor_id = r["correct_article_id"]

            if query_chunks is None or cor_id not in present_ids:
                continue

            # Query vector
            query_emb = mean_pool_query(query_chunks)

            # Candidate selection within same HC
            cand_index = np.where(hc_arr == hc)[0]
            if len(cand_index) == 0:
                continue

            # Cosine baseline scoring
            scores_cos = article_emb_matrix[cand_index] @ query_emb
            order_cos = cand_index[np.argsort(-scores_cos)]
            cand_ids_cos = emb_ids[order_cos]

            # Restrict to top-N cosine for reranking to simulate production
            top_indexes = order_cos[:min(top_n_cosine, len(order_cos))]

            # Build features and rerank
            feats = build_candidate_features(query_emb, article_emb_df, article_emb_matrix, top_indexes, hc)
            probs = self._model.predict_proba(feats)[:, 1]
            order_model = top_indexes[np.argsort(-probs)]
            cand_ids_model = emb_ids[order_model]

            # Compute metrics for both cosine & reranker
            rows.append(compute_metrics(cand_ids_cos, cand_ids_model, cor_id))

        if not rows:
            self.logger.warning("No evaluable feedback rows found.")
            return {}
        df = pd.DataFrame(rows)
        return df.mean().to_dict()

    def rerank_from_candidates(
            self,
            query_embedding: List[List[float]],
            help_center_id: str,
            candidate_ids: list[str],
            candidate_embeddings: list[np.ndarray],
    ) -> pd.DataFrame:
        """
        Given query embedding, help center ID, candidate article IDs and embeddings,
        return a DataFrame with article_id and predicted relevance scores.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Reranker model is not loaded or trained yet.")
        query_emb_pooled = np.asarray(mean_pool_query(query_embedding), dtype=np.float32)
        feats = build_candidate_features_inference(query_emb_pooled, candidate_embeddings, candidate_ids,
                                                   help_center_id)
        probs = self._model.predict_proba(feats)[:, 1]
        results = (
            pd.DataFrame({
                "article_id": candidate_ids,
                "score": probs
            })
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
        self.logger.debug(f"Reranked {len(candidate_ids)} candidates for HC {help_center_id}.")
        return results

    def save_artifacts(self, output_dir: str):
        """Save trained model with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(output_dir, f"reranker_{self.model_name}_{timestamp}.joblib")

        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self._model, model_path)
        self.logger.info(f"Model saved to {model_path}")

    def load_artifacts(self, artifacts_dir: str):
        """Load trained model"""
        model_files = sorted(Path(artifacts_dir).glob("reranker_*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No model artifacts found in {artifacts_dir}")
        latest_model_path = model_files[-1]

        self._model = joblib.load(latest_model_path)
        self._is_fitted = True
        self.logger.info(f"Model loaded from {latest_model_path}")
