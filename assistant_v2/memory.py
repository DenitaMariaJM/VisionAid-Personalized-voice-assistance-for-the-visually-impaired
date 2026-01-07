import faiss
import numpy as np
from openai import OpenAI

client = OpenAI()

DIM = 1536  # embedding size
index = faiss.IndexFlatL2(DIM)
texts = []

def get_embedding(text):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(emb.data[0].embedding).astype("float32")

def store_memory(text):
    vector = get_embedding(text)
    index.add(vector.reshape(1, -1))
    texts.append(text)

def search_memory(query, k=2):
    if len(texts) == 0:
        return []

    q_vec = get_embedding(query)
    _, indices = index.search(q_vec.reshape(1, -1), k)
    return [texts[i] for i in indices[0] if i < len(texts)]
