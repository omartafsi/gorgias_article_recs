from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field, validator


class RerankRequest(BaseModel):
    help_center_id: str = Field(..., description="Help center ID (e.g. '25171')")
    query_embedding: List[List[float]] = Field(..., description="Query embedding vector")
    top_k: int = Field(10, description="Number of candidates to retrieve from index")

    @validator("query_embedding")
    def validate_query_embedding(cls, v):
        if not v or not isinstance(v, list):
            raise ValueError("query_embedding must be a non-empty list of lists of floats.")
        if not all(isinstance(vec, list) and all(isinstance(x, (float, int)) for x in vec) for vec in v):
            raise ValueError("query_embedding must be a 2D list of numeric values.")
        return v
