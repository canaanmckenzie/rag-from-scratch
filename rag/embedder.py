import os
import re
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

#embedding model to map each chunk to point in R^d (d = 384 for MiniLLM)
model = SentenceTransformer('all-MiniLM-L6-v2')

#chunk sizes is approxmimately 500 tokens ~= 2000 characters
MAX_CHARS = 2000

#cleaning
def clean_text(text: str) -> str:
    """
    Normalize the input text by removing the extra whitespaces
    """
    text = re.sub(r'\s+',' ',text)
    return text.strip()

#chunking
def chunk_text(text: str, max_chars: int = MAX_CHARS) -> List[str]:
    """
    Splite the document into semantically clean, character-limited chunks
    c_i "chunks"
    """
    sentences = re.split(r'(?<=[.!?]) +', text) #naive sentence splitter
    chunks, current_chunk = [], ""

    for sentence in sentences:
        # append until the size limit is reached
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip()) #saves full chunk
            current_chunk = sentence #starts a new chunk
    if current_chunk:
        chunks.append(current_chunk.strip()) #catch the final piece
    return chunks

def embed_chunks(chunks: List[str]) -> np.ndarray:
    """
    SentenceTransformer to embed each chunk
    implements vec(c_i) = f_embed(c_i)
    """
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings)

def process_txt_file(path: Path) -> List[Dict]:
    """
    processes one .txt file into a list of embedded chunk records
    record:
    {
        'source': filename,
        'chunk_id': i,
        'text': original_text,
        'embedding': np.array([...])
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    clean = clean_text(raw_text)
    chunks = chunk_text(clean)

    #apply f_embed to every chunk
    embeddings = embed_chunks(chunks)

    return [
        {
            "source": path.name,
            "chunk_id": i,
            "text": chunk,
            "embedding": emb
        }
        for i, (chunk, emb) in enumerate(zip(chunks,embeddings))
    ]

def build_corpus(data_dir: str = "data/") -> List[Dict]:
    """
    Walk the data/ folder, process each .txt file into embedded chunks.
    returns the full corpus
    C = { (c_1, vec_1), (c_2, vec_2), ..., (c_n, vec_n) }
    """
    corpus = []
    for path in Path(data_dir).glob("*.txt"):
        corpus.extend(process_txt_file(path))
    return corpus

