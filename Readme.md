# 👗 Fashion Sence AI

## (Fashion Visual Search & Personalized Styling Assistant)

![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?logo=streamlit)
![HuggingFace](https://img.shields.io/badge/LLM-Gemma%203B-orange?logo=huggingface)
![FAISS](https://img.shields.io/badge/Search-FAISS-green?logo=facebook)
![CLIP](https://img.shields.io/badge/Image%20Encoder-CLIP-lightgrey?logo=openai)
![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A modern **AI-powered web application** that helps users **search for visually similar fashion products**, receive **outfit recommendations** based on their style preferences and **simulate personalized experiences** using recent trends and LLMs — all in one elegant interface built with **Streamlit**, **CLIP**, and **Gemma LLM**.

## 🌐 Website - [DEMO]()

![Fashion Assistant UI](Src/Main)

---

## 🧠 Project Overview

Traditional fashion search engines rely on tags or manual filters. Our system enhances user experience by allowing:

* **Image-based Search**: Upload any fashion item image to retrieve visually similar products.
* **Text-based Search**: Enter queries like *"oversized hoodie, floral print dress"* and retrieve matching products.
* **Personalized Suggestions**: Based on your browsing or simulated history.
* **Intelligent Outfit Completion**: Generate LLM-powered suggestions to complete your outfit using trending insights.

This app can serve as a **virtual fashion assistant**, a **personal stylist**, or a foundation for an e-commerce AI integration.

---

## 🚀 Features at a Glance

| Module                        | Description                                                           |
| ----------------------------- | --------------------------------------------------------------------- |
| 🔍 Visual Search              | Upload an image or type a query to find similar fashion products.     |
| 🧠 LLM-based Outfit Generator | Uses **Gemma 3B** via Hugging Face API to suggest outfit completions. |
| 👤 History Simulation         | Fake user history generation + product-based similarity suggestions.  |
| 📈 Fashion Trend Inference    | Extracts trending keywords from inventory and online data.            |
| 📊 FAISS Indexing             | Efficient nearest-neighbor search using pretrained embeddings.        |
| 🌐 Hugging Face Integration   | Seamless API token handling and LLM inference.                        |
| ✅ Modular Pipeline            | Each part of the pipeline is reusable, testable, and extensible.      |

---

## 🗂️ Project Structure

```bash
Fashion AI/
├── Assets/                  # FAISS Index, product IDs, and image embeddings
├── Data/                    # Cleaned CSV files for jeans and dresses
├── Modules/                 # Modular Python scripts for each major functionality
│   ├── dataloader.py
│   ├── faiss_index.py
│   ├── outfit_suggester.py
│   ├── preprocessing.py
│   ├── search.py
│   ├── trends.py
│   └── user_profile.py
├── Src/                     # Static assets like animation GIFs or logos
├── Notebooks/               # Python Jupyter Notebooks
├── Test_Images/             # Sample test images
├── app.py                   # Main Streamlit app file
├── requirements.txt         # Required Python libraries
└── README.md                # You're here!
```

---

## 🛠️ Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)
* **Image Embeddings**: [CLIP (ViT-L/14)](https://github.com/openai/CLIP)
* **Text Embeddings**: [SentenceTransformers](https://www.sbert.net/)
* **LLM Suggestion Engine**: [Gemma-3B](https://huggingface.co/google/gemma-1.1-2b-it) (via HF Inference API)
* **Nearest Neighbors Search**: [FAISS](https://github.com/facebookresearch/faiss)
* **Trends Analysis**: Web scraping + TF-IDF on product metadata
* **Deployment-ready**: Works on Streamlit Cloud

---

## 📦 Installation & Setup

> Recommended Python version: `>=3.11`

1. **Clone the repository**

```bash
git clone https://github.com/MohitGupta0123/Fashion-Sense-AI.git
cd Fashion-Sense-AI
```

2. **Install the dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the application**

```bash
streamlit run app.py
```

4. **Enter Hugging Face Token**

* Get your free token from: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
* Paste it in the **sidebar** to access outfit suggestions using the Gemma model.

---

## 🧭 How to Use

Once the app is running:

### 👤 Step-by-Step

1. **Upload** an image or enter a style description like `"striped top, loose fit"`.
2. Adjust the number of desired results using the slider.
3. View **Top Matching Products** — displayed with image and price.
4. Click **🧪 Simulate Fake History** to create browsing history (or use your own).
5. See **Suggestions Based on History**.
6. Tap **🧠 Generate Outfit** to let the LLM suggest fashion complements (shoes, accessories, layering items, etc.).

---

## 🖼️ UI Screenshots

| Visual Search                       | Personalized Recommendations      | Outfit Completion                       |
| ----------------------------------- | --------------------------------- | --------------------------------------- |
| ![Visual Search](Src/Animation.gif) | ![Suggestions](Src/Animation.gif) | ![Outfit Suggestion](Src/Animation.gif) |

---

## 🧪 Sample Use Cases

* 🔍 A user uploads a black denim jacket → finds 10 similar styles instantly.
* 🛍️ A shopper queries for `"Off Shoulder dresses"` → retrieves relevant dresses.
* 🤖 Fake user browsing history is generated → recommendations are shown.
* 🎨 LLM completes the outfit with accessories and layering items based on the uploaded look.

---

## 🔐 Hugging Face Token Handling

* The `HF_TOKEN` is required to query Gemma for outfit recommendations.
* It is securely saved in `st.session_state` for each user session.
* It is **never stored persistently** or sent elsewhere.

---

## 🎯 Future Improvements

* 🔒 Authenticated user sessions with persistent history
* 🛒 Add-to-cart integration for e-commerce use
* 🗣️ Voice-based outfit search (via Whisper or Speech-to-Text APIs)
* 🧵 Tailor chatbot integration for fashion Q\&A

---

## 👨‍💻 Author

**Mohit Gupta**
🏆 Kaggle, Hackathon & Research Enthusiast
🎓 [mgmohit1111@gmail.com](mailto:mgmohit1111@gmail.com)
🔗 [LinkedIn](https://linkedin.com/in/mohitgupta012) | [GitHub](https://github.com/MohitGupta0123)
