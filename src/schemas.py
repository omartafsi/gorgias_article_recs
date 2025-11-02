from pydantic import BaseModel
import pandas as pd
import json


class Article(BaseModel):
    id: str
    help_center_id: str
    title: str
    content: str

    @staticmethod
    def from_dataframe(dataframe: pd.DataFrame):
        return [
            Article(
                id=row["id"],
                help_center_id=row["help_center_id"],
                title=row["title"],
                content=row["content"],
            )
            for _, row in dataframe.iterrows()
        ]


class ArticleFeedback(BaseModel):
    query_embeddings: list[list[float]]
    help_center_id: str
    recommended_article_id: str
    correct_article_id: str

    @staticmethod
    def from_dataframe(dataframe: pd.DataFrame):
        return [
            ArticleFeedback(
                query_embeddings=json.loads(row["query_embeddings"]),
                help_center_id=row["help_center_id"],
                recommended_article_id=row["recommended_article_id"],
                correct_article_id=row["correct_article_id"],
            )
            for _, row in dataframe.iterrows()
        ]
