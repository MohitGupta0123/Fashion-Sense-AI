import os
import json
import requests
from Modules.user_profile import summarize_user_preferences

def _to_str(x):
    if x is None:
        return ""
    # convert lists/sets/tuples to comma-separated strings
    if isinstance(x, (list, tuple, set)):
        return ", ".join(map(str, x))
    return str(x)

def generate_outfit_gemma(
    image_url, row, user_id, df, user_history, trend_string,
    number_of_suggestions=5, hf_token=None, model_id="google/gemma-3-27b-it"
):
    """
    Return outfit suggestions using HF Router /v1/chat/completions with an image.
    """
    assert hf_token, "❌ HF_TOKEN must be provided."

    API_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    # Step 1: Summarize user preferences
    user_brands, user_styles, user_description = summarize_user_preferences(
        user_id, df, user_history, top_k=3
    )

    # Step 2: Build safe strings for the prompt
    product_name      = _to_str(row.get("product_name", ""))
    brand             = _to_str(row.get("brand", ""))
    style_attributes  = _to_str(row.get("style_attributes", ""))
    description       = _to_str(row.get("description", ""))
    price             = _to_str(row.get("selling_price", ""))

    prompt_text = f"""
Using the image below and the following product and user profile information,
suggest {number_of_suggestions} stylish outfit items to complete this look.

🎯 Product Details:
- Name: {product_name}
- Brand: {brand}
- Style Attributes: {style_attributes}
- Description: {description}
- Price: ₹{price}

🧍‍♀️ User Style Preferences:
- Favorite Brands: {user_brands}
- Preferred Style Features: {user_styles}
- Liked Descriptions: {user_description}

🔥 Trending Styles:
{_to_str(trend_string)}

💡 Provide creative, trendy outfit pieces. Use bullet points and add a short reason for each. Do not ask follow-up questions.
"""

    payload = {
        "model": model_id,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        # If the router returns an error payload, this will raise and we’ll show details.
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.HTTPError as http_err:
        # Surface router's error body to see why it rejected the request.
        try:
            err_txt = resp.text
        except Exception:
            err_txt = str(http_err)
        return f"❌ HTTP {resp.status_code} from HF router: {err_txt}"
    except Exception as e:
        return f"❌ Error generating outfit: {e}"
