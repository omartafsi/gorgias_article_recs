import os

HELP_CENTER_ID = "help_center_id"
QUERY_EMBEDDINGS_COL = "query_embeddings"
ARTICLE_ID = "article_id"
SCORE = "score"

ARTIFACTS_DIRECTORY = os.getenv("RERANKER_MODEL_PATH", "artefacts/")
EMB_PATH = os.getenv("ARTICLE_EMB_PARQUET", "results/article_embeddings.parquet")
FAISS_DIR = os.getenv("FAISS_DIR", "results/faiss_hc_indexes/")
