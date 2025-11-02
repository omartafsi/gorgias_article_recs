import json
import numpy as np
import pandas as pd
from pathlib import Path


class DataLoader:
    """Utility class for loading embeddings and feedback data_loader."""

    @staticmethod
    def load_embeddings(path: str) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
        """Loads article embeddings from a Parquet file."""
        p = Path(path)
        df = pd.read_parquet(p)
        df = df.astype({"id": str, "help_center_id": str})

        df["embedding"] = df["embedding"].apply(lambda x: np.asarray(x, dtype=np.float32))
        arr = np.stack(df["embedding"].to_list()).astype(np.float32)

        article_id_to_index = {id_: i for i, id_ in enumerate(df["id"])}
        return df, arr, article_id_to_index

    @staticmethod
    def load_feedback(path: str) -> pd.DataFrame:
        """Loads feedback interactions from a CSV file."""
        p = Path(path)
        df = pd.read_csv(
            p,
            dtype={
                "help_center_id": str,
                "recommended_article_id": str,
                "correct_article_id": str,
            },
        )

        def _parse_query(x):
            try:
                return json.loads(x) if isinstance(x, str) else x
            except Exception:
                return None

        df["query_embeddings_parsed"] = df["query_embeddings"].apply(_parse_query)
        return df
