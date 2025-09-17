# RecTask – Content-based Recommender with Sentence Embeddings

This repository provides a simple content-based recommendation system using sentence embeddings from `sentence-transformers/all-MPNet-base-v2`.

## Features:
- **Weighted Embeddings:** Combines embeddings from **Genres (0.6 weight)** and **Description (0.4 weight)**. The `Description` field often incorporates the `Title` if a separate `Description` is not available. Embeddings are L2 normalized per-item.
- **Cosine Similarity / ANN:** Uses cosine/inner-product similarity; automatically leverages FAISS ANN if available.
- **Dynamic Cache:** A cache file (e.g., `your_data_name_embedding_cache.npz`) is created *next to your input `--data` JSON file*. This cache stores per-item embeddings and hashes, skipping re-embedding for unchanged items.
- **Cache Invalidation:** The cache automatically invalidates when `Description`, `Genres`, model name, or embedding weights change.
- **Outputs:** Recommendations include Id, Title, Genres, and Score.
- **Evaluation Mode:** Includes a feature to evaluate the recommender's performance (Precision, Recall, NDCG) using historical user interactions.
- **Embedding Saving:** Option to save all computed embeddings to a `.npz` file.

## Setup:
1) **Python environment and dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   FAISS is optional and auto-detected. If you want ANN speed-ups, install `faiss-cpu`.

2) **Data:**
   Provide a JSON file similar to `all 1.json` (either a list of items or an object like `{"items": [...]}`) with the following keys:
   - `Id` (string/int) - *Required*
   - `Title` (string) - *Recommended, used as fallback for description*
   - `Genres` (list of dicts with `Name`, list of strings, or a single string) - *Recommended*
   - `Description` (string) - *Recommended, primary field for content description*

   For **evaluation mode**, you will also need an interactions CSV file:
   - No header.
   - Columns should be (in order): `profileid`, `contentid`, `contenttype`, `timestamp`, `episodecount`, `wathcedcount`, `applicationid`, `deviceid`, `lang`, `country`. Only `profileid` and `contentid` are used by the script.

## Run:

First, activate your virtual environment:
```bash
source .venv/bin/activate
```

Then, you can run the script in different modes. You can call either entrypoint:
- `python main.py ...` (wrapper)
- or `python recommend.py ...` (direct)

**1. Generate Recommendations:**
   ```bash
   python main.py --data "/path/to/all+1.json" --k 10
   ```
   - `Id` and `Title` for the default seed will be printed if not explicitly specified.

   **Optional Seed Item:**
   ```bash
   python main.py --data "/path/to/all+1.json" --seed 317111 --k 10
   ```

   **Optional: Save all computed embeddings:**
   ```bash
   python main.py --data "/path/to/all+1.json" --save "my_catalog_embeddings.npz"
   ```
   This will save a `.npz` file containing item IDs and their corresponding embeddings.

   **Optional: Model and weights**
   ```bash
   python main.py --data "/path/to/all+1.json" --model-name sentence-transformers/all-MPNet-base-v2 --genre-weight 0.6 --desc-weight 0.4
   ```

**2. Evaluate Recommender Performance (using interactions data):**
   ```bash
   python main.py --data "/path/to/all+1.json" --interactions "/path/to/interactions.csv" --k 10
   ```
   - This mode will load your catalog and interactions, then calculate Precision@k, Recall@k, and NDCG@k averaged across eligible users.
   - Eligible users are those with 5 or more interactions. One random interaction is chosen as the "seed" for recommendation, and the rest of the user's interactions are considered "relevant" for evaluation.
   - You can combine this with `--save` to save the embeddings after evaluation:
     ```bash
     python main.py --data "/path/to/all+1.json" --interactions "/path/to/interactions.csv" --k 10 --save "eval_embeddings.npz"
     ```

## ANN acceleration (FAISS)
- The script automatically builds and loads a FAISS index for fast top-k search. No extra CLI flags are required.
- Index type: IndexFlatIP on L2-normalized vectors (exact inner product). The index and IDs are persisted next to the embedding cache:
  - {json_base}_embedding_cache_ann.faiss
  - {json_base}_embedding_cache_ann_ids.npy
- It is used both for:
  - Single-seed recommend: python main.py --data "/path/to/all+1.json" --k 10
  - Evaluation: python main.py --data "/path/to/all+1.json" --interactions "/path/to/interactions.csv" --k 10
- If FAISS is unavailable, the script automatically falls back to exact similarity without the index.

---
