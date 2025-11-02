# Help center recommendation

## Project Description

The objective of the project is to build a help_center article recommender based on the client's message
It contains three major segment:
* An article_embedding_pipeline that generates article embedding based on their title
* A training pipeline for a re-ranker that takes message_query embedding, article embedding re-ranks articles
* An api that takes as input the query, help_center_id and outputs an ordered list of recommended articles
* The api uses faiss framework as a vector database and loads the model to re-rank the already ranked output
* The idea is to have two jobs, one for embedding generation and another of model training

## Getting Started

Installation
------------
    $ pip install -e .
    
Usage
------------

```python
import os
from sklearn.model_selection import train_test_split
import json
from article_embedding_pipeline.data_manager.loader import load_articles
from article_embedding_pipeline.embeddings.generator import generate_article_embeddings
from article_embedding_pipeline.embeddings.storage import save_embeddings
from reranker.reranker import Reranker
from reranker.data_loader.loader import DataLoader

embeddings_out = "./results/article_embeddings.parquet"
feedback_path = "./data/article_feedback.csv"
output_dir = "./artefacts"
# Generating article embeddings
articles = load_articles("./data/help_center_articles.csv")
ids, help_center_ids, embeddings = generate_article_embeddings(
    articles,
    batch_size=32,
    model_name="avsolatorio/GIST-small-Embedding-v0",
    use_gpu=False,
)
save_embeddings(ids, help_center_ids, embeddings, embeddings_out)

# training and evaluating reranker
emb_df, arr, id_to_index = DataLoader.load_embeddings(embeddings_out)
fb_df = DataLoader.load_feedback(feedback_path)
fb_train, fb_val = train_test_split(fb_df, test_size=0.2, random_state=42)
reranker = Reranker(model_name="logistic", n_negatives=3)
reranker.fit(emb_df=emb_df, arr=arr, fb_df=fb_train)

if fb_val is not None:
    metrics = reranker.evaluate(fb_val, emb_df, arr)
    print(json.dumps(metrics, indent=2))

os.makedirs(output_dir, exist_ok=True)
reranker.save_artifacts(output_dir)
```

From CLI: 
make sure to have feedback and article data available in data/
------------
    $ article_embedding_pipeline generate-embeddings --articles-csv ./data/help_center_articles.csv --embeddings-out ./results/article_embeddings.parquet  --faiss-index-out ./results/faiss_hc_indexes
    $ reranker train --embeddings-path ./results/article_embeddings.parquet --feedback-path ./data/article_feedback.csv  --output-dir ./artefacts
Run as a service
------------
    $ uvicorn api.main:app --reload --port 8080
    $ curl -X POST "http://localhost:8080/rerank" \                                                                                                                      
     -H "Content-Type: application/json" \
     -d '{
           "help_center_id": "25171", 
           "query_embedding": [[-0.014673777855932713 .. -0.012392970733344555]],
           "top_k": 5 }'


Project Structure
------------

    ├── README.md          <- README of the project.
    ├── data               <- Raw data.
    ├── src                <- source code for training and predicting contact reasons.
    ├── notebooks          <- Jupyter notebooks.
    ├── requirements.txt   <- The requirements file contains all the necessary libs to run the project.
    ├── tests              <- tests forlder.
    ├── api                <- to run the project as a service.
    ├── artefacts          <- saving reranker artefacts.
    ├── results            <- saving generated article embeddings and model artefacts.
    └── noxfile.py         <- black, build, tests.               

--------
