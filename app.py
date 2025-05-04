from rag.embedder import build_corpus
corpus = build_corpus("data/")
print(f"Loaded {len(corpus)} chunks.")
print(corpus[0]['text'])
print(corpus[0]['embedding'].shape)
