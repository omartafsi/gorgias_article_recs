from src.article_embedding_pipeline.ann.indexer import (
    load_faiss_index_for_hc,
    search,
)
import numpy as np
import faiss


def extract_candidates_from_faiss(request, faiss_dir: str):
    """
    Retrieve top-k candidate IDs, scores, and reconstruct their embeddings
    for a given help center from a FAISS IndexIDMap(IndexFlatIP).
    """
    hc_id = str(request.help_center_id)
    faiss_index = load_faiss_index_for_hc(hc_id, faiss_dir)
    query_emb_pooled = np.array(mean_pool_query(request.query_embedding), dtype=np.float32).reshape(1, -1)
    candidate_ids, _ = search(faiss_index, query_emb_pooled, top_k=request.top_k)
    candidate_embeddings = reconstruct_embeddings_from_faiss(faiss_index, candidate_ids, hc_id)

    return candidate_ids, candidate_embeddings


def reconstruct_embeddings_from_faiss(index, candidate_ids, hc_id: str):
    """
    Reconstruct candidate article embeddings using the article ids on which we built the index
    because IndexIDMap(IndexFlatIP) indexes don't have a builtin reconstruct
    """
    id_array = faiss.vector_to_array(index.id_map)
    base_index = index.index  # Inner IndexFlatIP
    candidate_embeddings = []

    for candidate_id in candidate_ids:
        matches = np.where(id_array == int(candidate_id))[0]
        if matches.size:
            # If the FAISS ID exists in the mapping, reconstruct its embedding from the inner index
            candidate_embeddings.append(base_index.reconstruct(int(matches[0])))
        else:
            print(f"Warning: FAISS ID {candidate_id} not found in index (HC {hc_id})")

    return candidate_embeddings


def mean_pool_query(chunks):
    # Ideally, we would have this function in a centralized repo that we import
    # and use the different jobs (training and serving)
    if chunks is None or len(chunks) == 0:
        return None
    a = np.array(chunks, dtype=np.float32)
    return a.mean(axis=0)
