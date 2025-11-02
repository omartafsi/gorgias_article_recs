import logging
import os
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def build_faiss_indexes(df: pd.DataFrame, output_dir: str) -> None:
    """
    Builds one FAISS index per help center and saves it to disk.
    Each FAISS index uses article IDs as FAISS IDs (via IndexIDMap).
    For simplicity of the pipeline, we'll assume that all ids are ints, type checks should be enforced in Prod.
    """

    os.makedirs(output_dir, exist_ok=True)
    unique_hcs = df["help_center_id"].unique()

    for hc in tqdm(unique_hcs, desc="Building HC indexes"):
        hc_df = df[df["help_center_id"] == hc]

        if len(hc_df) == 0:
            continue

        embeddings = np.vstack(hc_df["embedding"].to_list()).astype("float32")
        dim = embeddings.shape[1]

        # Create FAISS index with ID mapping
        base_index = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap(base_index)

        # Use numeric article IDs as FAISS IDs
        faiss_ids = np.array([int(aid) for aid in hc_df["id"]], dtype=np.int64)

        # Add embeddings with IDs
        index.add_with_ids(embeddings, faiss_ids)

        # Save to disk
        index_path = os.path.join(output_dir, f"hc_{hc}.faiss")
        faiss.write_index(index, index_path)

    LOGGER.info(f"Built {len(unique_hcs)} FAISS indexes successfully with article ID mapping.")


def load_faiss_index_for_hc(help_center_id: str, base_dir: str) -> faiss.Index:
    """
    Loads a specific help center's FAISS index.
    """
    index_path = os.path.join(base_dir, f"hc_{help_center_id}.faiss")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No FAISS index found for help center {help_center_id}")
    return faiss.read_index(index_path)


def search(index, query_embedding: np.ndarray, top_k=10) -> Tuple[List[str], List[float]]:
    """
    Perform a FAISS similarity search for a given query embedding with shape (1, 384)
    """
    query = np.asarray(query_embedding, dtype=np.float32)
    d, i = index.search(query, top_k)
    candidate_ids = [str(int(x)) for x in i[0] if x != -1]
    distances = d[0].tolist()
    return candidate_ids, distances
