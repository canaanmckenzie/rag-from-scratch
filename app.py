from rag.embedder import build_corpus
from rag.retriever import build_faiss_index, search

corpus = build_corpus("data/")
print(f"total chunks: {len(corpus)}") #log chunks
build_faiss_index(corpus)

query = "What rough beast slouches towards Bethlehem?"
results = search(query, k=3)

for r in results:
    print(f"\n Score: {r['score']:.4f} | Chunk ID: {r['chunk_id']} | File: {r['source']}")
    print(r['text'])

#print(f"Loaded {len(corpus)} chunks.")
#print(corpus[0]['text'])
#print(corpus[0]['embedding'].shape)
