#feature branch feat/retriever
#semantic search using FAISS based on cosine similarity

import os
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer

#use same embedding model used during indexing
#vector cability across chunk + query steps
model = SentenceTransformer('all-MiniLM-L6-v2')

INDEX_PATH = "embeddings/index.faiss"
META_PATH = "embeddings/meta.json"

#normalize embeddings to unit length
#cosine similiarty = inner product for normalized vectors
def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

#input corpus from embedder.py
#output index file + metadata file
def build_faiss_index(corpus: List[Dict]) -> None:
    vectors = np.array([entry["embedding"] for entry in corpus])
    vectors = normalize(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    Path("embeddings/").mkdir(exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    metadata = [
        {
            "text": entry["text"],
            "source": entry["source"],
            "chunk_id": entry["chunk_id"]
        }
        for entry in corpus
    ]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

#search FAISS index with an input query string
#returns top-k most similar chunks from the corpus
def search(query: str, k: int = 5) -> List[Dict]:
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("Index or metadata not found. Run build_faiss_index first")

    #vector index + metadata
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    #embed input query
    q_vec = model.encode([query])
    q_vec = normalize(q_vec)

    #clamp k to actual index size to avoid garbage
    actual_k = min(k, index.ntotal)
    if actual_k < k:
        print(f"[retriever] Requested top-{k}, but index has only {index.ntotal} vectors. Clamping to {actual_k}")
    
    #search with cosine singularity
    scores, indices = index.search(q_vec, actual_k)
    results = []
    for i, score in zip(indices[0], scores[0]):
        if i < 0 or score < -1e38:
            continue #skip placeholders
        hit = metadata[i]
        hit["score"] = float(score)
        results.append(hit)
    return results

