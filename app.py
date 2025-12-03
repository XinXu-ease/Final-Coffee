import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import predict_rating, predict_cluster, generate_flavor_profile, plot_pca_interactive, CLUSTER_NAMES, CLUSTER_DESCRIPTIONS
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Coffee ML App", layout="centered")

st.markdown("""
<style>

:root {
    --coffee-primary: #6F4E37;       /* 咖啡棕 */
    --coffee-secondary: #BFA58A;     /* 浅棕 */
    --coffee-light: #F7F3EF;         /* 奶白 */
    --text-dark: #3a3a3a;
    --text-light: #6e6e6e;
}

/* input box background */
input[type="text"],
input[type="number"],
textarea,
div[data-baseweb="select"] > div {
    background-color:var(--coffee-light) !important;        
    border: 1px solid var(--coffee-primary) !important;       
    border-radius: 10px !important;
    color: var(--coffee-primary) !important;                  
}

            /* Streamlit 的 label 实际上是放在一个 css-xxxyyy 的 div 里的标题 */
[data-testid="stWidgetLabel"] > div:nth-child(1) {
    font-weight: 800 !important;   /* 强制加粗 */
    color: #6F4E37 !important;      /* 咖啡色 */
    font-size: 1rem !important;     /* 字体微调（可选） */
}    

            
/*  hover */
input[type="text"]:hover,s
input[type="number"]:hover,
textarea:hover,
div[data-baseweb="select"] > div:hover {
    border: 1px solid #A08979 !important;       
}
     
/* title color */
h1, h2, h3, h4 {
    color: var(--coffee-primary) !important;
}


/* default button */
.stButton button {
    background-color:var(--coffee-primary) !important; 
    color: white !important;
    font-weight: 700 !important;           
    border-radius: 6px !important;
    padding: 0.6em 1em !important;
    border: none !important;
}

/* hover  */
.stButton button:hover {
    background-color: #5a3f2c !important;  
    color: var(--coffee-light) !important;             
}
            /* ===== 精准改 number_input 右侧加减区域底色 ===== */

/* 包住 - / + 的那一整块背景 */
div[data-testid="stNumberInput"] > div:nth-child(2) {
    background-color: #F3F2F7 !important;   /* 改成和输入框一样的浅紫/米色 */
    border-radius: 0 10px 10px 0 !important;
}

/* 按钮本身透明，只显示文字 */
div[data-testid="stNumberInput"] button {
    background-color: transparent !important;
    box-shadow: none !important;
    color: #3A3A3A !important;             /* - 和 + 的颜色 */
}
</style>
""", unsafe_allow_html=True)

st.title("☕ Coffee Flavor & Rating Predictor")
st.write("Input the coffee information, I will predict the rating and determine which flavor cluster it belongs to.")

# Data load
df = pd.read_csv("data/df_for_pca.csv")

# =======================
#   User Inputs
# =======================

st.header("📥 Input the coffee information")

# Name
st.text_input("☕ Optional: Coffee Name", key="coffee_name")

# -------- OPTIONAL: Auto-fill from coffee name --------
if st.session_state.get("coffee_name"):
    name = st.session_state["coffee_name"]

    matches = df[df["name"].str.contains(name, case=False, na=False)]

    if len(matches) > 0:
        coffee_row = matches.iloc[0]   

        # autofill session_state
        st.session_state["auto_roast"] = coffee_row["roast"]
        st.session_state["auto_loc"] = coffee_row["loc_country"]
        st.session_state["auto_price"] = coffee_row["100g_USD"]
        st.session_state["auto_rating"] = coffee_row["rating"]


# Description
text = st.text_area(
    "Description",
    value=st.session_state.get(
        "default_text",
        "High-toned, richly sweet. Strawberry guava, roasted cacao nib, lime zest, jasmine, almond in aroma and cup. Sweetly-tart structure with juicy, balanced acidity; plush, syrupy mouthfeel. The finish is long and resonant, with lime zest and strawberry guava in the short, rounding to cocoa-toned jasmine in the deeply sweet long."
    ),
    key="desc_input"
)

# Price
price = st.number_input("Price（100g/USD）", min_value=0.0, value=st.session_state.get("auto_price", 10.0))
if "auto_price" in st.session_state:
    price = st.session_state["auto_price"]

# Roast
roast = st.selectbox(
    "Roast",
    ["Light", "Medium-Light", "Medium", "Medium-Dark", "Dark"],
    index = ["Light", "Medium-Light", "Medium", "Medium-Dark", "Dark"].index(
        st.session_state.get("auto_roast", "Light")
    ),
)
if "auto_roast" in st.session_state:
    roast = st.session_state["auto_roast"]

# Country
loc_list = ['Hong Kong', 'United States', 'Canada', 'Taiwan', "Hawai'i", 'Australia'
 'England', 'Uganda', 'Mexico', 'Belgium', 'United States And Floyd',
 'Guatemala', 'Japan', 'Peru', 'Honduras' ,'China', 'Kenya', 'Malaysia',
 'New Taiwan']
loc = st.selectbox(
    "Location_Country",
     loc_list,
 index = loc_list.index( st.session_state.get("auto_loc", "Hong Kong") )
)
if "auto_loc" in st.session_state:
    loc = st.session_state["auto_loc"]

# Optional rating
user_rating = st.text_input(
    "Optional: Rating",
    value=str(st.session_state.get("auto_rating", "")),
)
if user_rating.strip() != "":
    rating_used = float(user_rating)
elif "auto_rating" in st.session_state:
    rating_used = float(st.session_state["auto_rating"])
else:
    rating_used = predict_rating(text, roast, loc, price)

# =======================
#   Predict Button
# =======================

from utils import (
    predict_cluster,
    predict_rating,
    get_user_pca_point
)

# Load data
pca_data = np.load("models/pca_data.npy")
cluster_labels = np.load("models/cluster_labels.npy")
cluster_labels = np.array(cluster_labels, dtype=int)
print(cluster_labels[:20])

# ---------------------------
# 1. PREDICT BUTTON LOGIC
# ---------------------------
if st.button("Predict"):

    # --- 1. Predict Rating ---
    #  Determine rating (use user provided or model)
    if user_rating.strip() != "":
        predicted_rating = float(user_rating)   # use user input
    elif "auto_rating" in st.session_state:
        predicted_rating = float(st.session_state["auto_rating"])  # use autofill
    else:
        predicted_rating = predict_rating(text, roast, loc, price)  # fallback to model

    # --- 2. Predict Cluster ---
    cluster_id = predict_cluster(text, roast, loc, price, predicted_rating)
    cluster_name = CLUSTER_NAMES[cluster_id]

    # --- 3. PCA User Point ---
    user_xy = get_user_pca_point(text, roast, loc, price, predicted_rating)

    # --- Save to Session State ---
    st.session_state["predicted_rating"] = predicted_rating
    st.session_state["cluster_id"] = cluster_id
    st.session_state["cluster_name"] = cluster_name
    st.session_state["user_xy"] = user_xy


# ---------------------------
# 2. ALWAYS DISPLAY RESULTS IF THEY EXIST
# ---------------------------
if "predicted_rating" in st.session_state:

    st.subheader("⭐ Rating")
    st.write(f"**{st.session_state['predicted_rating']:.2f} / 100**")

    st.subheader("🗂️ Cluster")
    st.write(f"**{st.session_state['cluster_name']}**")
    cluster_desc = CLUSTER_DESCRIPTIONS.get(st.session_state["cluster_id"], "")
    st.markdown(
    f"<p style='color:#6e6e6e;'>{cluster_desc}</p>",
    unsafe_allow_html=True
    )

    # --- Plot PCA figure ---
    fig = plot_pca_interactive(
        pca_data, 
        cluster_labels,
        df,
        CLUSTER_NAMES,
        st.session_state["user_xy"]
    )
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("🌍 Cluster-wise Global Distribution")

    # Initialize selected cluster once
    if "selected_cluster" not in st.session_state:
        st.session_state["selected_cluster"] = st.session_state["cluster_id"]

    cluster_names = CLUSTER_NAMES

    # ---- Buttons Row 1 ----
    c1, c2, c3 = st.columns(3)
    if c1.button(cluster_names[0]):
        st.session_state["selected_cluster"] = 0
    if c2.button(cluster_names[1]):
        st.session_state["selected_cluster"] = 1
    if c3.button(cluster_names[2]):
        st.session_state["selected_cluster"] = 2

    # ---- Buttons Row 2 ----
    c4, c5, c6 = st.columns(3)
    if c4.button(cluster_names[3]):
        st.session_state["selected_cluster"] = 3
    if c5.button(cluster_names[4]):
        st.session_state["selected_cluster"] = 4
    if c6.button(cluster_names[5]):
        st.session_state["selected_cluster"] = 5

    # Use selected state
    selected_cluster = st.session_state["selected_cluster"]

    # ---- Filter df by cluster ----
    df_cluster = df[df["Cluster"] == selected_cluster]

    if len(df_cluster) == 0:
        st.warning("No country data for this cluster.")
    else:
        df_country_cluster = df_cluster.groupby("loc_country").agg(
            avg_rating=("rating", "mean"),
            avg_price=("100g_USD", "mean"),
            count=("rating", "count"),
        ).reset_index()

        fig_cluster_map = px.choropleth(
            df_country_cluster,
            locations="loc_country",
            locationmode="country names",
            color="count",
            hover_name="loc_country",
            hover_data={
                "count": True,
                "avg_rating": ":.2f",
                "avg_price": ":.2f",
            },
            color_continuous_scale="YlOrRd",
            title=f" {cluster_names[selected_cluster]} — Global Distribution",
        )

        fig_cluster_map.update_traces(
            marker_line_width=0.2,
            marker_line_color="black"
        )
        fig_cluster_map.update_geos(
            showcoastlines=True,
            coastlinecolor="black",
            coastlinewidth=0.3, 
            showcountries=True,
            countrycolor="black",
            countrywidth=0.2,   
            showland=True,
            landcolor="white",
            showlakes=False,
            showframe=False,
        )

        fig_cluster_map.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            coloraxis_colorbar=dict(
                title="Count",
                thickness=12,
                len=0.4,
                bgcolor="rgba(255,255,255,0.7)"
            )
        )

        st.plotly_chart(fig_cluster_map, use_container_width=True)



# ---------------------------
# 3. GENERATE AI ANALYSIS BUTTON
# ---------------------------
if "cluster_id" in st.session_state:

    if st.button("Generate AI Analysis"):
        flavor_desc = generate_flavor_profile(
            text=text,
            cluster_id=st.session_state["cluster_id"],
            roast=roast,
            loc=loc,
            price=price,
        )

        st.subheader("💡 LLM Flavor Description")
        st.write(flavor_desc)