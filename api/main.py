import traceback
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException

from api.constants import ARTIFACTS_DIRECTORY, FAISS_DIR
from api.request import RerankRequest
from api.response import RerankResponse
from api.utils import extract_candidates_from_faiss
from reranker.reranker import Reranker

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Article Reranker API",
    description="Retrieve and rerank help center articles using FAISS and a trained logistic model.",
    version="v1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
reranker = Reranker()


@app.on_event("startup")
async def startup_event():
    global reranker

    reranker.load_artifacts(ARTIFACTS_DIRECTORY)


@app.get("/")
def root():
    return {"message": "Reranker API is running"}


@app.post("/rerank", response_model=RerankResponse, status_code=200)
async def rerank(request: RerankRequest) -> RerankResponse:
    try:
        global reranker

        hc_id = str(request.help_center_id)
        # Retrieve candidates from vector db
        candidate_ids, candidate_embeddings = extract_candidates_from_faiss(
            request=request,
            faiss_dir=FAISS_DIR,
        )
        # Rerank candidates using the trained model
        rerank_df = (
            reranker.rerank_from_candidates(
                query_embedding=request.query_embedding,
                help_center_id=hc_id,
                candidate_embeddings=candidate_embeddings,
                candidate_ids=candidate_ids
            )
            if candidate_ids
            else None
        )
        return RerankResponse.from_rerank_results(
            candidate_ids=candidate_ids,
            reranked_df=rerank_df
        )

    except Exception as e:
        logger.exception("Error during /rerank")
        raise HTTPException(status_code=500, detail="Internal error")
