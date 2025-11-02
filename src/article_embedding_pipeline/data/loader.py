import pandas as pd
from schemas import Article


def load_articles(articles_csv_path: str) -> list[Article]:
    df = pd.read_csv(articles_csv_path)
    df = df.drop_duplicates(
        subset=["id", "help_center_id", "title", "content"],
        keep="first",
    ).reset_index(drop=True)
    return Article.from_dataframe(df)
