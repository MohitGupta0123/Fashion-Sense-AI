import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
# from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# Load models once globally
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
clip_model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

# clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)

text_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_image_embedding(image_path: str) -> np.ndarray:
    """
    Generate image embedding from a given image path using CLIP.

    Returns:
        np.ndarray: L2-normalized image embedding (768D)
    """
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
            return torch.nn.functional.normalize(features, p=2, dim=-1).cpu().numpy()[0]
    except Exception as e:
        print(f"❌ Image error: {image_path} — {e}")
        return None

def get_text_embedding(text: str) -> np.ndarray:
    """
    Generate a text embedding using SentenceTransformer.

    Returns:
        np.ndarray: Text embedding (384D)
    """
    try:
        return text_model.encode(text, show_progress_bar=False)
    except Exception as e:
        print(f"❌ Text error: {text[:60]}... — {e}")
        return None

def generate_all_image_embeddings(df, image_base_dir: str) -> dict:
    """
    Generate image embeddings for all product_ids.

    Returns:
        dict: {product_id: embedding}
    """
    image_embeddings = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Image Embeddings"):
        pid = row["product_id"]
        image_path = os.path.join(image_base_dir, f"{pid}.jpg")
        emb = get_image_embedding(image_path)
        if emb is not None:
            image_embeddings[pid] = emb
    return image_embeddings

def generate_all_text_embeddings(df, text_inputs: list[str]) -> dict:
    """
    Generate text embeddings for all rows in df.

    Args:
        df: DataFrame with product_id column
        text_inputs: List of combined text fields

    Returns:
        dict: {product_id: embedding}
    """
    text_embeddings = {}
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Text Embeddings")):
        pid = row["product_id"]
        emb = get_text_embedding(text_inputs[i])
        if emb is not None:
            text_embeddings[pid] = emb
    return text_embeddings

def combine_embeddings(image_embeddings: dict, text_embeddings: dict, product_ids: list) -> dict:
    """
    Combine image and text embeddings into one vector.

    Returns:
        dict: {product_id: [image + text] embedding}
    """
    combined = {}
    for pid in product_ids:
        img = image_embeddings.get(pid)
        txt = text_embeddings.get(pid)
        if img is not None and txt is not None:
            combined[pid] = np.concatenate([img, txt])
    return combined


# from Modules.embedding import generate_all_image_embeddings, generate_all_text_embeddings, combine_embeddings
# from Modules.preprocessing import prepare_text_for_embedding

# text_inputs = prepare_text_for_embedding(df)

# image_embeddings = generate_all_image_embeddings(df, "/kaggle/input/dataset-ecomerce/Images/Images")
# text_embeddings = generate_all_text_embeddings(df, text_inputs)

# combined_embeddings = combine_embeddings(image_embeddings, text_embeddings, df["product_id"].tolist())