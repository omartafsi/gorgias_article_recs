from __future__ import annotations
import pandas as pd
from pydantic import BaseModel, Field


class RerankResponse(BaseModel):
    ranked_articles: list[str] = Field(default_factory=list, description="Ranked article IDs")
    message: str = Field(default="", description="Status or error message")

    @classmethod
    def from_rerank_results(
            cls,
            candidate_ids: list[str],
            reranked_df: pd.DataFrame | None,
    ) -> "RerankResponse":
        if reranked_df is None or reranked_df.empty:
            return RerankResponse(
                ranked_articles=candidate_ids,
                message=f"Reranking skipped — returning FAISS candidates",
            )

        return RerankResponse(ranked_articles=reranked_df.sort_values("score", ascending=False)["article_id"].tolist())
