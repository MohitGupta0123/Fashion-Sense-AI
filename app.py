import streamlit as st
import pandas as pd
from PIL import Image
import tempfile
import random
import requests

from Modules.dataloader import load_csvs, verify_column_match, merge_datasets, clean_price_fields, filter_columns
from Modules.preprocessing import fill_missing_fields
from Modules.faiss_index import load_faiss_assets
from Modules.search import search_similar
from Modules.outfit_suggester import generate_outfit_gemma
from Modules.user_profile import summarize_user_preferences
from Modules.trends import get_combined_trend_string

# --- CONFIG ---
st.set_page_config(page_title="👗 Fashion Assistant", layout="wide")
st.title("👗 Fashion Sense AI")

st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🔐 Hugging Face Token")
    hf_token_input = st.text_input("Enter your HF_TOKEN", type="password")
    if st.button("🔓 Submit"):
        st.session_state["HF_TOKEN"] = hf_token_input
        st.success("HF_TOKEN saved in session.")

    st.markdown("### ℹ️ How to Use")
    st.markdown("""
    1. **Upload** a clothing image *or* enter a style query.
    2. Set the **number of similar results** to retrieve.
    3. View **Top Matching Products**.
    4. Click **Simulate Fake History** to see personalized suggestions.
    5. Generate **Outfit Completion Suggestions** using LLM.
    """)
    
    st.caption("Developed by Mohit Gupta")

# --- TOKEN CHECK PAGE CONTENT ---
if not st.session_state.get("HF_TOKEN"):
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 👋 Welcome to Fashion Visual Assistant")
        st.markdown("""
        This AI-powered assistant helps you:
        - 👗 Search visually similar fashion products.
        - 🧠 Get outfit recommendations using LLMs.
        - 📈 Discover trends and style suggestions personalized to your taste.

        👉 (\*_\*) **Please enter your** **`HF_TOKEN`** **in the sidebar to get started!**
        """)
    with col2:
        st.image("Src/Animation.gif", width=250)
    st.stop()

# --- LOAD DATA ---
@st.cache_resource
def load_assets(hf_token):
    dress, jeans = load_csvs(
        "Data/dresses_bd_processed_data.csv",
        "Data/jeans_bd_processed_data.csv"
    )
    assert verify_column_match(dress, jeans), "Column mismatch in dress and jeans data."

    df = merge_datasets(dress, jeans)
    df = clean_price_fields(df)
    df = filter_columns(df)
    df = fill_missing_fields(df)

    faiss_index, product_ids, _ = load_faiss_assets("Assets")
    trend_string = get_combined_trend_string(df, use_internet=True, hf_token=hf_token)

    return df, faiss_index, product_ids, trend_string

# --- REQUIRE TOKEN TO LOAD DATA ---
df, faiss_index, product_ids, trend_string = load_assets(st.session_state["HF_TOKEN"])

# --- SESSION STATE ---
if "user_id" not in st.session_state:
    st.session_state.user_id = "user123"
if "user_history" not in st.session_state:
    st.session_state.user_history = {}

user_id = st.session_state.user_id
user_history = st.session_state.user_history

# --- INPUT SECTION ---
st.markdown("## 🛍️ Search Your Style")
uploaded_file = st.file_uploader("📄 Upload a clothing image", type=["jpg", "jpeg", "png"])
text_query = st.text_input("🎯 Enter style query (e.g. 'floral, oversized')", "")
top_k = st.number_input("🔢 Number of similar results (you want to see and write in multiple of 5)", min_value=1, max_value=30, value=15, step=1)

# --- PREPARE STYLING ---
st.markdown("""
    <style>
    .product-card {
        text-align: center;
        padding: 10px;
        height: 340px;
        overflow: hidden;
        border: 1px solid #eee;
        border-radius: 10px;
    }
    .product-img {
        height: 200px;
        object-fit: cover;
        margin-bottom: 5px;
        border-radius: 6px;
    }
    .caption {
        font-size: 14px;
        line-height: 1.3em;
        max-height: 3.8em;
    }
    </style>
""", unsafe_allow_html=True)

# --- SEARCH ---
st.markdown("## 🔍 Top Matching Products")
top_ids = []
temp_image_path = None
if uploaded_file or text_query.strip():
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            temp_image_path = tmp.name
        st.image(temp_image_path, caption="📸 Uploaded Image", width=300)

    top_ids = search_similar(faiss_index, product_ids, temp_image_path, text_query, top_k)

    cols = st.columns(5)
    for i, pid in enumerate(top_ids):
        row = df[df["product_id"] == pid].iloc[0]
        with cols[i % 5]:
            st.markdown(f"""
                <div class="product-card">
                    <img src=\"{row['feature_image_s3']}\" class="product-img" />
                    <div class="caption">{row['product_name']}<br/>₹{row['selling_price']}</div>
                </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# --- USER HISTORY SIMULATION ---
if st.button("🧪 Simulate Fake History"):
    random_ids = df.sample(top_k)["product_id"].tolist()
    combined_ids = list(set(random_ids + top_ids))
    user_history[user_id] = combined_ids
    st.success("✅ Fake history created using random and visually similar products.")

    st.markdown("### 🌐 Fake History Products")
    cols3 = st.columns(5)
    for i, pid in enumerate(combined_ids):
        row = df[df["product_id"] == pid].iloc[0]
        with cols3[i % 5]:
            st.markdown(f"""
                <div class="product-card">
                    <img src=\"{row['feature_image_s3']}\" class="product-img" />
                    <div class="caption">{row['product_name']}<br/>₹{row['selling_price']}</div>
                </div>
            """, unsafe_allow_html=True)

# --- SUGGESTIONS BASED ON HISTORY ---
st.markdown("## 👤 Suggestions Based on User History")
if user_id in user_history and user_history[user_id]:
    history_ids = user_history[user_id]
    all_similar = []
    for pid in history_ids:
        image_url = df[df["product_id"] == pid]["feature_image_s3"].values[0]
        try:
            image_content = requests.get(image_url).content
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(image_content)
                local_path = tmp.name
            similar = search_similar(faiss_index, product_ids, local_path, "", 3)
            all_similar.extend(similar)
        except Exception as e:
            st.warning(f"Skipping image {image_url} due to error: {e}")

    suggestion_ids = list(set(all_similar) - set(history_ids))[:top_k]

    if suggestion_ids:
        cols2 = st.columns(5)
        for i, pid in enumerate(suggestion_ids):
            row = df[df["product_id"] == pid].iloc[0]
            with cols2[i % 5]:
                st.markdown(f"""
                    <div class="product-card">
                        <img src=\"{row['feature_image_s3']}\" class="product-img" />
                        <div class="caption">{row['product_name']}<br/>₹{row['selling_price']}</div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No new suggestions found based on history.")

st.markdown("---")

# --- OUTFIT COMPLETION ---
st.markdown("## 💡 Outfit Completion Suggestions")
if user_history.get(user_id):
    reference_id = top_ids[0] if top_ids else user_history[user_id][0]
    top_row = df[df["product_id"] == reference_id].iloc[0]
    image_url = top_row["feature_image_s3"]

    if st.button("🧠 Generate Outfit"):
        suggestions = generate_outfit_gemma(
            image_url=image_url,
            row=top_row,
            user_id=user_id,
            df=df,
            user_history=user_history,
            trend_string=trend_string,
            number_of_suggestions=5,
            hf_token=st.session_state["HF_TOKEN"]
        )
        st.markdown(suggestions)