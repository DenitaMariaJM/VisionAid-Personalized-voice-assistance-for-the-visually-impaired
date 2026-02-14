"""In-memory semantic memory backed by FAISS embeddings."""

import sqlite3
import threading
import logging
from datetime import datetime, timezone

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
text_timestamps = []
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

    try:
        # Call OpenAI embeddings API
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=trimmed
        )
        # Convert embedding to NumPy float32 array
        # FAISS requires float32 vectors
        return np.array(emb.data[0].embedding).astype("float32")
    except Exception as exc:
        logger.warning("embedding_failed error=%s", exc)
        return None


def _utc_now_text():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _format_semantic_entry(text, created_at):
    stamp = created_at or "unknown-time"
    return f"[semantic @ {stamp}] {text}"


def _format_user_fact_entry(text, created_at):
    stamp = created_at or "unknown-time"
    return f"[user fact @ {stamp}] {text}"


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
    if vector is None:
        return

    # Add the vector to the FAISS index
    # Reshape is required: (1, DIM)
    created_at = _utc_now_text()
    with _lock:
        index.add(vector.reshape(1, -1))
        # Store the original text
        texts.append(trimmed)
        text_timestamps.append(created_at)

    if MEMORY_PERSIST:
        try:
            conn = sqlite3.connect(DB_NAME)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO semantic_memory (text, embedding, created_at) VALUES (?, ?, ?)",
                (trimmed, vector.tobytes(), created_at),
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
    if q_vec is None:
        return []

    # Perform similarity search in FAISS index
    # Returns distances and indices
    with _lock:
        _, indices = index.search(q_vec.reshape(1, -1), k)
        # Retrieve corresponding texts using indices
        results = []
        for i in indices[0]:
            if 0 <= i < len(texts):
                created_at = text_timestamps[i] if i < len(text_timestamps) else None
                results.append(_format_semantic_entry(texts[i], created_at))
        return results


def load_memory():
    if not MEMORY_PERSIST:
        return
    try:
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("SELECT text, embedding, created_at FROM semantic_memory ORDER BY id ASC")
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
        text_timestamps.clear()
        for text, blob, created_at in rows:
            if not blob:
                continue
            vec = np.frombuffer(blob, dtype="float32")
            if vec.size != DIM:
                continue
            index.add(vec.reshape(1, -1))
            texts.append(text)
            text_timestamps.append(created_at)


def clear_memory():
    with _lock:
        index.reset()
        texts.clear()
        text_timestamps.clear()

    if MEMORY_PERSIST:
        try:
            conn = sqlite3.connect(DB_NAME)
            cur = conn.cursor()
            cur.execute("DELETE FROM semantic_memory")
            conn.commit()
            conn.close()
        except Exception:
            logger.warning("memory_clear_failed")


def list_memory(limit=3):
    if limit <= 0:
        return []
    with _lock:
        pairs = list(zip(texts, text_timestamps))
        return [_format_semantic_entry(text, created_at) for text, created_at in pairs[-limit:]]


def store_user_fact(text: str) -> bool:
    if not text:
        return False

    trimmed = _trim_text(text, MEMORY_SNIPPET_CHARS)
    vector = get_embedding(trimmed)
    if vector is None:
        return False

    created_at = _utc_now_text()
    try:
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM user_facts WHERE LOWER(text) = LOWER(?) LIMIT 1",
            (trimmed,),
        )
        if cur.fetchone():
            conn.close()
            return False
        cur.execute(
            "INSERT INTO user_facts (text, embedding, created_at) VALUES (?, ?, ?)",
            (trimmed, vector.tobytes(), created_at),
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        logger.warning("user_fact_store_failed")
        return False


def search_user_facts(query: str, k: int = 2) -> list[str]:
    if k <= 0:
        return []

    try:
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT text, embedding, created_at
            FROM user_facts
            ORDER BY id DESC
            LIMIT 200
            """
        )
        rows = cur.fetchall()
        conn.close()
    except Exception:
        logger.warning("user_fact_search_failed")
        return []

    if not rows:
        return []

    query_text = (query or "").strip()
    if not query_text:
        return [_format_user_fact_entry(text, created_at) for text, _, created_at in rows[:k]]

    q_vec = get_embedding(query_text)
    if q_vec is None:
        return [_format_user_fact_entry(text, created_at) for text, _, created_at in rows[:k]]

    q_norm = float(np.linalg.norm(q_vec))
    if q_norm == 0:
        return [_format_user_fact_entry(text, created_at) for text, _, created_at in rows[:k]]

    scored = []
    for text, blob, created_at in rows:
        if not text or not blob:
            continue
        try:
            vec = np.frombuffer(blob, dtype="float32")
            if vec.size != q_vec.size:
                continue
            denom = float(np.linalg.norm(vec) * q_norm)
            if denom == 0:
                continue
            similarity = float(np.dot(vec, q_vec) / denom)
            scored.append((similarity, text, created_at))
        except Exception:
            continue

    if not scored:
        return [_format_user_fact_entry(text, created_at) for text, _, created_at in rows[:k]]

    scored.sort(key=lambda item: item[0], reverse=True)
    return [_format_user_fact_entry(text, created_at) for _, text, created_at in scored[:k]]
