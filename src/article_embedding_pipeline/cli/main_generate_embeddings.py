import click

from article_embedding_pipeline.data_manager.loader import load_articles
from article_embedding_pipeline.embeddings.generator import generate_article_embeddings
from article_embedding_pipeline.embeddings.storage import save_embeddings
from article_embedding_pipeline.ann.indexer import build_faiss_indexes


@click.command()
@click.option("--articles-csv", type=str, required=True)
@click.option("--embeddings-out", type=str, required=True)
@click.option("--faiss-index-out", type=str, required=True)
@click.option("--batch-size", type=int, default=32, required=False)
@click.option("--model-name", type=str, default="avsolatorio/GIST-small-Embedding-v0", required=False)
@click.option("--use-gpu", is_flag=True, default=False, required=False)
def generate_embeddings(
    articles_csv: str,
    embeddings_out: str,
    faiss_index_out: str,
    batch_size: int = 32,
    model_name: str = "avsolatorio/GIST-small-Embedding-v0",
    use_gpu: bool = False,
) -> None:
    click.echo("Loading articles...")
    articles = load_articles(articles_csv)

    click.echo("Generating embeddings...")
    ids, help_center_ids, embeddings = generate_article_embeddings(
        articles,
        batch_size=batch_size,
        model_name=model_name,
        use_gpu=use_gpu,
    )

    click.echo("Saving embeddings...")

    df = save_embeddings(ids, help_center_ids, embeddings, embeddings_out)

    click.echo("Building FAISS index...")
    build_faiss_indexes(df, faiss_index_out)

    click.echo(f"Done. {len(df)} embeddings indexed.")


if __name__ == "__main__":
    generate_embeddings()
