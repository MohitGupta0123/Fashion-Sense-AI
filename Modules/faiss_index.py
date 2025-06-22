import faiss
import numpy as np
import os
import pickle

def build_faiss_index(combined_embeddings: dict) -> tuple[faiss.IndexFlatL2, list]:
    """
    Build a FAISS L2 index from combined embeddings.

    Args:
        combined_embeddings (dict): {product_id: embedding}

    Returns:
        Tuple of (FAISS index, list of product IDs in index order)
    """
    ids = list(combined_embeddings.keys())
    vectors = np.stack([combined_embeddings[pid] for pid in ids]).astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    return index, ids

def save_faiss_assets(index: faiss.IndexFlatL2, combined_embeddings: dict, save_dir: str = "Assets"):
    """
    Save FAISS index and product ID order.

    Args:
        index: FAISS index to save
        combined_embeddings: Dictionary of product embeddings
        save_dir: Output directory to store index and metadata
    """
    os.makedirs(save_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(save_dir, "faiss_index.index"))

    with open(os.path.join(save_dir, "product_ids.pkl"), "wb") as f:
        pickle.dump(list(combined_embeddings.keys()), f)

    np.save(os.path.join(save_dir, "combined_vectors.npy"), np.stack([combined_embeddings[pid] for pid in combined_embeddings]))

def load_faiss_assets(load_dir: str = "Assets") -> tuple[faiss.IndexFlatL2, list, np.ndarray]:
    """
    Load FAISS index and metadata.

    Args:
        load_dir: Directory from which to load assets

    Returns:
        Tuple: (index, list of product_ids, combined_vectors array)
    """
    index = faiss.read_index(os.path.join(load_dir, "faiss_index.index"))

    with open(os.path.join(load_dir, "product_ids.pkl"), "rb") as f:
        ids = pickle.load(f)

    vectors = np.load(os.path.join(load_dir, "combined_vectors.npy"))

    return index, ids, vectors

def search_index(index: faiss.IndexFlatL2, query_vector: np.ndarray, top_k: int = 5) -> list[int]:
    """
    Perform a top-k similarity search on the FAISS index.

    Args:
        index: FAISS index
        query_vector: Combined image + text vector (shape: [1408] or [1, 1408])
        top_k: Number of top results to retrieve

    Returns:
        List of index positions of top_k most similar vectors
    """
    if query_vector.ndim == 1:
        query_vector = query_vector[np.newaxis, :]
    distances, indices = index.search(query_vector.astype("float32"), top_k)
    return indices[0].tolist()

# from modules.faiss_index import build_faiss_index, save_faiss_assets, load_faiss_assets, search_index

# # Build and save index
# faiss_index, ids = build_faiss_index(combined_embeddings)
# save_faiss_assets(faiss_index, combined_embeddings)

# # Later: Load index and perform search
# faiss_index, ids, _ = load_faiss_assets()

# # Example query
# top_indices = search_index(faiss_index, combined_embeddings[ids[0]], top_k=5)
# print("Top similar indices:", top_indices)