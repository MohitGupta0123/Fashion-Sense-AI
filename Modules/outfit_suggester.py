import os
import json
import requests
from Modules.user_profile import summarize_user_preferences

def generate_outfit_gemma(image_url, row, user_id, df, user_history, trend_string, number_of_suggestions=5, hf_token=None):
    """
    Generate outfit suggestions using HF API + image_url.

    Args:
        image_url (str): Public URL of the product image
        row (pd.Series): Product metadata
        user_id (str): Unique user identifier
        df (pd.DataFrame): Product dataframe
        user_history (dict): Dictionary of user browsing history
        trend_string (str): Current trending fashion keywords
        number_of_suggestions (int): Number of items to suggest
        hf_token (str): Hugging Face access token

    Returns:
        str: Generated fashion outfit suggestions
    """

    assert hf_token is not None, "❌ HF_TOKEN must be provided."
    
    print('Generating Outfit suggestions')

    API_URL = "https://router.huggingface.co/featherless-ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {hf_token}"}

    # Step 1: Get user preference summaries
    user_brands, user_styles, user_description = summarize_user_preferences(user_id, df, user_history, top_k=3)

    # Step 2: Construct prompt
    prompt_text = f"""
Using the image below and the following product and user profile information,
suggest {number_of_suggestions} stylish outfit items to complete this look.

🎯 **Product Details**:
- **Name**: {row['product_name']}
- **Brand**: {row['brand']}
- **Style Attributes**: {row['style_attributes']}
- **Description**: {row['description']}
- **Price**: ₹{row['selling_price']}

🧍‍♀️ **User Style Preferences**:
- **Favorite Brands**: {user_brands}
- **Preferred Style Features**: {user_styles}
- **Liked Descriptions**: {user_description}

🔥 **Trending Styles**:
{trend_string}

💡 Provide creative, trendy outfit pieces. Use bullet points and add a short reason for each. Do not ask follow-up questions.
"""

    payload = {
        "model": "google/gemma-3-12b-it",
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

    # Step 3: Call the API
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        print('Outfit suggestions generated successfully')
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error generating outfit: {str(e)}"

