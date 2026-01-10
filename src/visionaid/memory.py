"""In-memory semantic memory backed by FAISS embeddings."""

import sqlite3
import threading
import logging

import faiss               # Library for fast vector similarity search
import numpy as np         # Numerical operations for embeddings
from openai import OpenAI  # OpenAI client for generating embeddings

from .config import MEMORY_PERSIST, MEMORY_SNIPPET_CHARS, MEMORY_STORE_ASSISTANT
from .db import DB_NAME


# ==============================
# INITIAL SETUP
# ==============================

# Initialize OpenAI client
client = OpenAI()
logger = logging.getLogger(__name__)

# Dimensionality of the embedding vector
# Must match the embedding model used
DIM = 1536  # text-embedding-3-small output size

# Create a FAISS index for L2 (Euclidean) similarity search
# This index runs entirely on CPU (Raspberry Pi friendly)
index = faiss.IndexFlatL2(DIM)

# List to store the original text corresponding to each embedding
# The index position matches the FAISS vector index
texts = []
_lock = threading.Lock()


# ==============================
# EMBEDDING GENERATION
# ==============================

def _trim_text(text, max_chars):
    trimmed = text.strip() if text else ""
    if len(trimmed) <= max_chars:
        return trimmed
    return trimmed[:max_chars].rstrip() + "..."


def get_embedding(text):
    """
    Converts input text into a numerical vector (embedding)
    using an OpenAI embedding model.

    Parameters:
        text (str): Input text to be embedded

    Returns:
        numpy.ndarray: 1536-dimensional float32 embedding vector
    """

    trimmed = _trim_text(text, MEMORY_SNIPPET_CHARS)

    # Call OpenAI embeddings API
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=trimmed
    )

    # Convert embedding to NumPy float32 array
    # FAISS requires float32 vectors
    return np.array(emb.data[0].embedding).astype("float32")


# ==============================
# MEMORY STORAGE
# ==============================

def build_memory_entry(user_text, assistant_text=None):
    if not user_text:
        return ""
    if MEMORY_STORE_ASSISTANT and assistant_text:
        entry = f"User: {user_text}\nAssistant: {assistant_text}"
    else:
        entry = user_text
    return _trim_text(entry, MEMORY_SNIPPET_CHARS)


def store_memory(text):
    """
    Stores a piece of text into vector memory.

    Steps:
    1. Convert text into an embedding
    2. Add the embedding to the FAISS index
    3. Store the original text for retrieval

    Parameters:
        text (str): Text to be remembered
    """

    if not text:
        return
    trimmed = _trim_text(text, MEMORY_SNIPPET_CHARS)

    # Generate embedding for the text
    vector = get_embedding(trimmed)

    # Add the vector to the FAISS index
    # Reshape is required: (1, DIM)
    with _lock:
        index.add(vector.reshape(1, -1))
        # Store the original text
        texts.append(trimmed)

    if MEMORY_PERSIST:
        try:
            conn = sqlite3.connect(DB_NAME)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO memory (text, embedding) VALUES (?, ?)",
                (trimmed, vector.tobytes()),
            )
            conn.commit()
            conn.close()
        except Exception:
            logger.warning("memory_persist_failed")


# ==============================
# MEMORY RETRIEVAL
# ==============================

def search_memory(query, k=2):
    """
    Searches vector memory for texts that are semantically
    similar to the given query.

    Parameters:
        query (str): User query text
        k (int): Number of similar memories to retrieve

    Returns:
        list[str]: List of relevant past texts
    """

    # If no memory exists, return empty list
    if len(texts) == 0:
        return []

    # Cap k to available memories to avoid invalid indices
    k = min(k, len(texts))
    if k <= 0:
        return []

    # Convert query into embedding
    q_vec = get_embedding(query)

    # Perform similarity search in FAISS index
    # Returns distances and indices
    with _lock:
        _, indices = index.search(q_vec.reshape(1, -1), k)
        # Retrieve corresponding texts using indices
        return [texts[i] for i in indices[0] if 0 <= i < len(texts)]


def load_memory():
    if not MEMORY_PERSIST:
        return
    try:
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("SELECT text, embedding FROM memory ORDER BY id ASC")
        rows = cur.fetchall()
        conn.close()
    except Exception:
        logger.warning("memory_load_failed")
        return

    if not rows:
        return

    with _lock:
        index.reset()
        texts.clear()
        for text, blob in rows:
            if not blob:
                continue
            vec = np.frombuffer(blob, dtype="float32")
            if vec.size != DIM:
                continue
            index.add(vec.reshape(1, -1))
            texts.append(text)
