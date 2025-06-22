import pandas as pd
import os

def load_csvs(dress_path: str, jeans_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dress and jeans datasets from CSV files.

    Args:
        dress_path (str): Path to the dresses CSV file.
        jeans_path (str): Path to the jeans CSV file.

    Returns:
        Tuple of pandas DataFrames: (dress, jeans)
    """
    dress = pd.read_csv(dress_path)
    jeans = pd.read_csv(jeans_path)
    return dress, jeans

def verify_column_match(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """
    Check if two DataFrames have the same column structure.

    Args:
        df1, df2: DataFrames to compare.

    Returns:
        bool: True if columns match, False otherwise.
    """
    return df1.columns.to_list() == df2.columns.to_list()

def merge_datasets(dress: pd.DataFrame, jeans: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate two DataFrames into one.

    Args:
        dress: Dress DataFrame
        jeans: Jeans DataFrame

    Returns:
        Merged DataFrame
    """
    return pd.concat([dress, jeans], ignore_index=True)

def clean_price_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numerical prices from stringified dictionaries.

    Args:
        df: Input DataFrame with 'selling_price' and 'mrp' columns.

    Returns:
        Updated DataFrame with cleaned price columns.
    """
    df["selling_price"] = df["selling_price"].apply(lambda x: eval(x).get("INR") or eval(x).get("USD"))
    df["mrp"] = df["mrp"].apply(lambda x: eval(x).get("INR") or eval(x).get("USD"))
    return df

def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the relevant columns for downstream processing.

    Args:
        df: Full product DataFrame.

    Returns:
        Filtered DataFrame with selected columns.
    """
    return df[[
        "product_id", "feature_image_s3", "product_name", "brand",
        "description", "category_id", "style_attributes", "mrp",
        "selling_price", "meta_info"
    ]]


# from modules.dataloader import load_csvs, verify_column_match, merge_datasets, clean_price_fields, filter_columns

# dress, jeans = load_csvs(
#     "/kaggle/input/dataset-ecomerce/dresses_bd_processed_data.csv",
#     "/kaggle/input/dataset-ecomerce/jeans_bd_processed_data.csv"
# )

# if verify_column_match(dress, jeans):
#     df = merge_datasets(dress, jeans)
#     df = clean_price_fields(df)
#     df = filter_columns(df)
# else:
#     raise ValueError("❌ Mismatch in column structure between dress and jeans datasets.")
