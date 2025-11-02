import os
import click
from sklearn.model_selection import train_test_split
import json

from reranker.reranker import Reranker
from reranker.data_loader.loader import DataLoader


@click.command()
@click.option("--embeddings-path", type=str, required=True, help="Path to article embeddings parquet file.")
@click.option("--feedback-path", type=str, required=True, help="Path to feedback CSV file.")
@click.option("--output-dir", type=str, required=True, help="Directory to save model artifacts.")
@click.option("--test-size", type=float, default=0.2, show_default=True, help="Validation split ratio.")
@click.option("--random-state", type=int, default=42, show_default=True, help="Random seed.")
@click.option("--n-negatives", type=int, default=3, show_default=True, help="Number of random negatives per query.")
@click.option("--score/--no-score", default=True, show_default=True, help="Whether to compute validation metrics.")
def train(
    embeddings_path: str,
    feedback_path: str,
    output_dir: str,
    test_size: float,
    random_state: int,
    n_negatives: int,
    score: bool,
):
    # Important, I've verified that the query vectors given are l2 normalized
    # We'll assume that to be the default format for queries for the rest of the project

    emb_df, arr, id_to_index = DataLoader.load_embeddings(embeddings_path)
    fb_df = DataLoader.load_feedback(feedback_path)

    fb_train, fb_val = train_test_split(fb_df, test_size=test_size, random_state=random_state)

    reranker = Reranker(model_name="logistic", n_negatives=n_negatives)

    reranker.fit(emb_df=emb_df, arr=arr, fb_df=fb_train)

    if score and fb_val is not None:
        metrics = reranker.evaluate(fb_val, emb_df, arr)
        click.echo("Metric comparaison between cosine similarity only ranking and cosine + model_reranking")
        click.echo(json.dumps(metrics, indent=2))

    os.makedirs(output_dir, exist_ok=True)
    reranker.save_artifacts(output_dir)


if __name__ == "__main__":
    train()
