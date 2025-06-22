import os
import pickle
import numpy as np
from datetime import datetime

def save_pickle(obj, filepath):
    """
    Save any Python object to a .pkl file.
    """
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    log(f"✅ Saved pickle: {filepath}")

def load_pickle(filepath):
    """
    Load object from a .pkl file.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)

def save_numpy(arr, filepath):
    """
    Save a NumPy array to a .npy file.
    """
    np.save(filepath, arr)
    log(f"✅ Saved NumPy array: {filepath}")

def load_numpy(filepath):
    """
    Load a NumPy array from a .npy file.
    """
    return np.load(filepath)

def resolve_image_path(product_id: str, image_folder: str) -> str:
    """
    Construct full path to image using product_id.

    Args:
        product_id (str): Unique product ID
        image_folder (str): Directory where product images are stored

    Returns:
        str: Full path to product image
    """
    return os.path.join(image_folder, f"{product_id}.jpg")

def log(message: str):
    """
    Print a log message with a timestamp.

    Args:
        message (str): Message to print
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


# from modules.utils import save_pickle, load_pickle, resolve_image_path, log

# # Save product IDs list
# save_pickle(product_ids, "Assets/product_ids.pkl")

# # Load image path for display
# img_path = resolve_image_path("12345", "data/Images")

# # Log something
# log("Embeddings generated successfully.")