import os
import numpy as np
from PIL import Image as PILImage
import torch
from Modules.embedding import clip_model, clip_processor, text_model
from Modules.faiss_index import search_index

# Use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"

def encode_image(image_path: str) -> np.ndarray:
    """
    Encode an image using CLIP to get its 768D embedding.

    Args:
        image_path (str): Path to the image.

    Returns:
        np.ndarray: Image embedding.
    """
    try:
        image = PILImage.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
            emb = torch.nn.functional.normalize(features, p=2, dim=-1).cpu().numpy()[0]
        return emb
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {e}")

def encode_text(text_query: str) -> np.ndarray:
    """
    Encode a text query using SentenceTransformer.

    Args:
        text_query (str): Input query string.

    Returns:
        np.ndarray: Text embedding.
    """
    return text_model.encode(text_query, show_progress_bar=False)

def search_similar(
    faiss_index,
    product_ids: list,
    query_image_path: str = None,
    query_text: str = None,
    top_k: int = 5
) -> list[str]:
    """
    Perform hybrid visual + textual similarity search.

    Args:
        faiss_index: Loaded FAISS index
        product_ids: List of product_ids corresponding to FAISS order
        query_image_path: Optional path to input image
        query_text: Optional text query
        top_k: Number of results to return

    Returns:
        List of product_ids ranked by similarity
    """
    # Encode image
    if query_image_path:
        image_embedding = encode_image(query_image_path)
    else:
        image_embedding = np.zeros(512, dtype=np.float32)

    # Encode text
    if query_text:
        text_embedding = encode_text(query_text)
    else:
        text_embedding = np.zeros(384, dtype=np.float32)

    # Combine embeddings
    query_vector = np.concatenate([image_embedding, text_embedding]).astype("float32")

    # Validate input dimension
    if query_vector.shape[0] != faiss_index.d:
        raise ValueError(f"Query vector dim {query_vector.shape[0]} does not match FAISS index dim {faiss_index.d}")

    # Search in FAISS
    top_indices = search_index(faiss_index, query_vector, top_k=top_k)

    # Return product_ids of top results
    return [product_ids[i] for i in top_indices if i < len(product_ids)]


# from modules.search import search_similar

# # search using image + optional text
# top_product_ids = search_similar(
#     faiss_index,
#     product_ids=ids,
#     query_image_path="path/to/query.jpg",
#     query_text="mini dress with floral pattern",
#     top_k=5
# )

# # display product details
# for pid in top_product_ids:
#     row = df[df["product_id"] == pid].iloc[0]
#     print(row["product_name"], row["brand"], row["selling_price"])