import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix, hstack
import os
from openai import OpenAI
import plotly.express as px

# Load models
rf_model = joblib.load("models/rf_rating.pkl")
kmeans = joblib.load("models/kmeans.pkl")
tfidf = joblib.load("models/tfidf.pkl")

# Load preprocessors
scaler_price = joblib.load("models/rf_price_scaler.pkl")
rating_ohe_cols = joblib.load("models/rf_cat_cols.pkl")
ohe_cols = joblib.load("models/kmeans_ohe_cols.pkl")
scaler_cluster = joblib.load("models/scaler.pkl")

# SBERT
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Rating Features ----
def encode_for_rating(text, roast, loc, price):
    text_vec = tfidf.transform([text])

    # One-hot
    ohe = np.zeros(len(rating_ohe_cols))
    for i, col in enumerate(rating_ohe_cols):
        if col == f"roast_{roast}" or col == f"loc_country_{loc}":
            ohe[i] = 1
    ohe = ohe.reshape(1, -1)

    # Numeric
    price_arr = csr_matrix([[price]])

    # Combine
    X = hstack([text_vec, ohe, price_arr])
    return X

def predict_rating(text, roast, loc, price):
    X = encode_for_rating(text, roast, loc, price)
    rating = rf_model.predict(X)[0]
    return float(rating)


# ---- Cluster Features ----
def encode_for_cluster(text, roast, loc, price, rating):
    emb = sbert.encode([text])[0]
    ohe = np.zeros(len(ohe_cols))
    for i, col in enumerate(ohe_cols):
        if col == roast or col == loc:
            ohe[i] = 1
    # numeric = (price, rating) — scaled together
    nums_scaled = scaler_cluster.transform([[price, rating]])[0]

    X = np.hstack([emb, ohe, nums_scaled])
    return X.reshape(1, -1)

def predict_cluster(text, roast, loc, price, rating):
    X = encode_for_cluster(text, roast, loc, price, rating)
    cluster_id = int(kmeans.predict(X)[0])
    return cluster_id

# Cluster Data
CLUSTER_NAMES = {
    0: "Classic Cocoa & Nut",
    1: "Taiwanese Bright Fruit",
    2: "Geisha & Kona Elite",
    3: "US Specialty: Balanced",
    4: "Ultra-Luxury Capsules",
    5: "Third Wave: Tart"
}

CLUSTER_DESCRIPTIONS = {
    0: "The most affordable, classic-profile American coffee, defined by cocoa and nut notes with a Medium or Medium-Light roast.",
    1: "Dominated by high-rated Taiwanese roasters like Kakalove, this cluster is known for its bright, sweet, fruit-forward flavors and Medium-Light roast.",
    2: "The luxury cluster, featuring expensive and sought-after varietals like Geisha and Kona, which command very high prices and ratings.",
    3: "The largest cluster, representing the mainstream US specialty market with high-value, sweet, and balanced coffees almost exclusively Medium-Light roasted.",
    4: "An ultra-luxury niche, consisting of extremely rare coffees like Kopi Luwak and Geisha packaged into Nespresso-compatible capsules.",
    5: "Represents the 'Third Wave' coffee style, focusing on Light roasts to produce distinctly bright, tart, and vibrant flavor profiles."
}


def get_user_pca_point(text, roast, loc, price, rating):
    pca = joblib.load("models/pca_2d.pkl")
    X = encode_for_cluster(text, roast, loc, price, rating)  
    pca_xy = pca.transform(X)  
    if pca_xy.shape[1] == 1:
        return np.array([pca_xy[0, 0], 0.0])
    return pca_xy[0]
  
def plot_pca_interactive(pca_data, cluster_labels, df, CLUSTER_NAMES, user_xy=None):
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    print("DEBUG SHAPES:",
      pca_data.shape,
      cluster_labels.shape,
      len(df))

    cluster_labels = cluster_labels.astype(int)
    df_plot = pd.DataFrame({
        "PC1": pca_data[:, 0],
        "PC2": pca_data[:, 1],
        "Cluster": cluster_labels,
        "Cluster Name": [CLUSTER_NAMES[c] for c in cluster_labels],
        "Roast": df["roast"],
        "Origin": df["loc_country"],
        "Price": df["100g_USD"],
        "Rating": df["rating"],
        "Roaster": df["roaster"]
    })

    fig = px.scatter(
        df_plot,
        x="PC1",
        y="PC2",
        color="Cluster Name",
        hover_data=["Roast", "Origin", "Price", "Rating", "Roaster"],
        title="PCA Visualization of Coffee Clusters",
        opacity=0.75,
        render_mode="svg"
    )

    if user_xy is not None:
        fig.add_trace(go.Scatter(
            x=[user_xy[0]],
            y=[user_xy[1]],
            mode="markers+text",
            marker=dict(
                size=10,
                color="black",
                line=dict(width=2, color="yellow")  
            ),
            text=["Your Coffee"],
            textposition="top center",
            name="Your Coffee",  
            showlegend=True,
        ))
    fig.update_layout(
        legend_title_text="Cluster Name",
        showlegend=True
    )
    return fig


# ---- LLM Description ----
import json
cluster_keywords = json.load(open("data/cluster_keywords.json", "r"))

def generate_flavor_profile(text, cluster_id, roast, loc, price):
    api_key = os.getenv("LITELLM_TOKEN")
    if not api_key:
        raise ValueError("LITELLM_TOKEN not found. Please set it first.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://litellm.oit.duke.edu/v1"
    )

    cluster_name = CLUSTER_NAMES[cluster_id]
    keywords = cluster_keywords[str(cluster_id)]
    prompt = f"""
You are a professional coffee taster holding an SCA certification. Based on the information below, 
generate a professional yet easy-to-understand flavor description.

[User’s Original Notes]
{text}

[Cluster Name]
{cluster_name}

[Cluster Flavor Keywords]
{', '.join(keywords)}

[Coffee Parameters]
Roast Level: {roast}
Origin Country: {loc}
Price (per 100g USD): {price}

Please generate:
- A 3–5 sentence tasting summary covering aroma, mouthfeel, acidity, and overall flavor structure.
- Do NOT repeat what the user already wrote.
- The description must sound natural, coherent, and professionally written—not like stitched machine output.
- Respond in ENGLISH ONLY.
"""

    response = client.chat.completions.create(
        model="GPT 4.1 Mini",   
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content