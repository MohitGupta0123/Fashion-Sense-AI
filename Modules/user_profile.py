import pandas as pd

def clean_style_attr(style_attr) -> str:
    """
    Standardize style_attributes field.

    Args:
        style_attr (dict or str or NaN)

    Returns:
        str: Cleaned style string
    """
    if isinstance(style_attr, dict):
        return ", ".join(f"{k}: {v}" for k, v in style_attr.items())
    elif isinstance(style_attr, str):
        return style_attr.strip()
    else:
        return "Unknown"

def summarize_user_preferences(user_id: str, df: pd.DataFrame, history_dict: dict, top_k: int = 5):
    """
    Generate a summary of user’s fashion preferences.

    Args:
        user_id (str): Unique user ID
        df (pd.DataFrame): Product metadata
        history_dict (dict): Dict containing product_id lists per user
        top_k (int): Number of top brands/styles to return

    Returns:
        Tuple[str, str, str]: (top brands, top styles, sample description text)
    """
    # Get product_ids user has interacted with
    pids = history_dict.get(user_id, [])
    rows = df[df["product_id"].isin(pids)]

    if rows.empty:
        return "No Brands", "No Styles", "No Description"

    # Top brands
    brands = rows["brand"].dropna().astype(str).value_counts().index.tolist()[:top_k]

    # Top style attributes
    styles_cleaned = rows["style_attributes"].apply(clean_style_attr)
    styles = styles_cleaned.value_counts().index.tolist()[:top_k]

    # Description summary (meta_info)
    descriptions = rows["meta_info"].dropna().astype(str).tolist()
    summary = " ".join(descriptions[:top_k * 2]) if descriptions else "No Description"

    return ", ".join(brands), ", ".join(styles), summary


# from modules.user_profile import summarize_user_preferences

# brands, styles, summary = summarize_user_preferences("user123", df, user_history, top_k=5)

# print("👕 Brands:", brands)
# print("🎨 Styles:", styles)
# print("📝 Summary:", summary)