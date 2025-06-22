# import os
# import json
# import requests
# from bs4 import BeautifulSoup

# API_URL = "https://router.huggingface.co/featherless-ai/v1/chat/completions"

# def extract_trend_keywords_with_gemma(description_text, hf_token):
#     headers = {
#         "Authorization": f"Bearer {hf_token}",
#     }

#     prompt = (
#         "From the text below, extract the top 50 trending fashion-related keywords, "
#         "such as types of clothing, styles, fabrics, silhouettes, patterns, or adjectives. "
#         "Avoid brand names. Only include words that appear in the input. Do not hallucinate. "
#         "Return a clean comma-separated list in English.\n\n"
#         f"{description_text}"
#     )

#     payload = {
#         "model": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
#         # "model": "google/gemma-3-12b-it",
#         "stream": False,
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [{"type": "text", "text": prompt}]
#             }
#         ]
#     }

#     try:
#         response = requests.post(API_URL, headers=headers, json=payload)
#         print("Response: ", response)
#         response.raise_for_status()
#         data = response.json()
#         print(data)
#         if "choices" in data and data["choices"]:
#             return data["choices"][0]["message"]["content"].strip()
#         else:
#             print("⚠️ Unexpected response:", data)
#             return ""
#     except Exception as e:
#         print(f"❌ API Error: {e}")
#         return ""


# def scrape_product_names(url, max_items=1000):
#     """
#     Scrape product titles from a public fashion site.
#     """
#     headers = {"User-Agent": "Mozilla/5.0"}
#     try:
#         response = requests.get(url, headers=headers, timeout=100)
#         soup = BeautifulSoup(response.content, "html.parser")

#         # Try both class types to ensure compatibility
#         name_divs = soup.find_all("div", class_="product-grids__copy-item") or soup.find_all("div", class_="product-name")
#         names = [div.get_text(strip=True) for div in name_divs]

#         print(f"✅ Scraped {len(names)} product names from {url}")
#         return names[:max_items]
#     except Exception as e:
#         print(f"❌ Web scrape failed: {e}")
#         return []


# def get_combined_trend_string(df, use_internet=True, max_desc=100, hf_token=None):
#     """
#     Combine local + web trend keywords using the language model.
#     """
#     assert hf_token, "HF_TOKEN is required for keyword extraction."

#     def limit_words(text, max_words=50):
#         return " ".join(text.split()[:max_words])

#     # --- 🧵 Local Trends ---
#     descriptions = df["description"].dropna().astype(str).tolist()
#     limited_desc = [limit_words(desc) for desc in descriptions[:max_desc]]
#     local_input = "\n".join(limited_desc)
#     local_trends = extract_trend_keywords_with_gemma(local_input, hf_token)
#     print("🧵 Local Trends:", local_trends)

#     # --- 🌐 Web Trends ---
#     if use_internet:
#         url = "https://www.fwrd.com/fw/content/products/lazyLoadProductsForward?currentPlpUrl=https%3A%2F%2Fwww.fwrd.com%2Ffwpage%2Fcategory-clothing%2F3699fc%2F&currentPageSortBy=featuredF&useLargerImages=false&outfitViewSession=false&showBagSize=false&lookfwrd=false&backinstock=false&preorder=false&_=1749445996960"
#         product_titles = scrape_product_names(url)
#         if product_titles:
#             web_input = "\n".join(product_titles)
#             web_trends = extract_trend_keywords_with_gemma(web_input, hf_token)
#             print("🌐 Web Trends:", web_trends)
#         else:
#             web_trends = ""
#             print("🌐 Web Trends: Not found.")
#     else:
#         web_trends = ""
#         print("🌐 Web Trends disabled.")

#     # --- Combine ---
#     combined_set = set(local_trends.split(",") + web_trends.split(","))
#     combined_keywords = sorted({kw.strip() for kw in combined_set if kw.strip()})
#     return ", ".join(combined_keywords)


# # from modules.trends import get_combined_trend_string

# # trend_string = get_combined_trend_string(df, use_internet=True)
# # print("🔥 Final Trend Keywords:\n", trend_string)


import pickle
import os

def get_combined_trend_string(df=None, use_internet=False, max_desc=100, hf_token=None):
    """
    Load precomputed trend keywords from Assets/trend_string.pkl.
    """
    trend_path = "Assets/trend_string.pkl"
    if os.path.exists(trend_path):
        with open(trend_path, "rb") as f:
            trend_string = pickle.load(f)
        print("✅ Loaded trend_string from file.")
        return trend_string
    else:
        print("❌ trend_string.pkl not found in Assets/. Please generate and save it first.")
        return ""
