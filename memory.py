import faiss               # Library for fast vector similarity search
import numpy as np         # Numerical operations for embeddings
from openai import OpenAI  # OpenAI client for generating embeddings


# ==============================
# INITIAL SETUP
# ==============================

# Initialize OpenAI client
client = OpenAI()

# Dimensionality of the embedding vector
# Must match the embedding model used
DIM = 1536  # text-embedding-3-small output size

# Create a FAISS index for L2 (Euclidean) similarity search
# This index runs entirely on CPU (Raspberry Pi friendly)
index = faiss.IndexFlatL2(DIM)

# List to store the original text corresponding to each embedding
# The index position matches the FAISS vector index
texts = []


# ==============================
# EMBEDDING GENERATION
# ==============================

def get_embedding(text):
    """
    Converts input text into a numerical vector (embedding)
    using an OpenAI embedding model.

    Parameters:
        text (str): Input text to be embedded

    Returns:
        numpy.ndarray: 1536-dimensional float32 embedding vector
    """

    # Call OpenAI embeddings API
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    # Convert embedding to NumPy float32 array
    # FAISS requires float32 vectors
    return np.array(emb.data[0].embedding).astype("float32")


# ==============================
# MEMORY STORAGE
# ==============================

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

    # Generate embedding for the text
    vector = get_embedding(text)

    # Add the vector to the FAISS index
    # Reshape is required: (1, DIM)
    index.add(vector.reshape(1, -1))

    # Store the original text
    texts.append(text)


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

    # Convert query into embedding
    q_vec = get_embedding(query)

    # Perform similarity search in FAISS index
    # Returns distances and indices
    _, indices = index.search(q_vec.reshape(1, -1), k)

    # Retrieve corresponding texts using indices
    return [texts[i] for i in indices[0] if i < len(texts)]
