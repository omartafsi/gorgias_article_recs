from article_embedding_pipeline.models.embedder import EmbeddingModel
from article_embedding_pipeline.data.preprocessing import preprocess_titles


def generate_article_embeddings(articles, batch_size: int, model_name: str, use_gpu: bool):
    embedder = EmbeddingModel(model_name=model_name, use_gpu=use_gpu)
    titles = preprocess_titles(articles)
    embeddings = embedder.encode(titles, batch_size=batch_size)
    ids = [article.id for article in articles]
    help_center_ids = [article.help_center_id for article in articles]
    return ids, help_center_ids, embeddings
