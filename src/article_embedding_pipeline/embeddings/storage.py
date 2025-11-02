import pandas as pd


def save_embeddings(ids, help_center_ids, embeddings, article_embedding_path):
    df = pd.DataFrame({
        "id": ids,
        "help_center_id": help_center_ids,
        "embedding": embeddings.tolist()
    })
    # For simplicity, we will store a dataset in parquet format for training
    df.to_parquet(article_embedding_path, index=False)
    return df