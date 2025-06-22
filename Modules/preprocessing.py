import pandas as pd
import numpy as np

def limit_words(text: str, max_words: int = 50) -> str:
    """
    Limit a given text to the first `max_words` words.

    Args:
        text (str): Input text string.
        max_words (int): Maximum number of words to keep.

    Returns:
        str: Truncated text string.
    """
    return " ".join(text.split()[:max_words])

def clean_style_attr(style_attr) -> str:
    """
    Clean the 'style_attributes' column to ensure it's in readable string format.

    Args:
        style_attr (str or dict): Raw style attribute.

    Returns:
        str: Cleaned style attribute string.
    """
    if isinstance(style_attr, dict):
        return ", ".join(f"{k}: {v}" for k, v in style_attr.items())
    elif isinstance(style_attr, str):
        return style_attr.strip()
    else:
        return "Unknown"

def fill_missing_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and clean textual columns.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df["description"] = df["description"].fillna("No description")
    df["meta_info"] = df["meta_info"].fillna("Unknown")
    df["style_attributes"] = df["style_attributes"].apply(clean_style_attr)
    return df

def prepare_text_for_embedding(df: pd.DataFrame) -> list[str]:
    """
    Combine text fields into a list of strings for text embedding.

    Args:
        df (pd.DataFrame): Product metadata DataFrame.

    Returns:
        list[str]: Combined text inputs for SentenceTransformer.
    """
    texts = []
    for _, row in df.iterrows():
        combined = f"{row['product_name']} {row['description']} {row['meta_info']} {row['style_attributes']}"
        texts.append(combined)
    return texts

# from modules.preprocessing import fill_missing_fields, prepare_text_for_embedding

# df = fill_missing_fields(df)
# text_inputs = prepare_text_for_embedding(df)
