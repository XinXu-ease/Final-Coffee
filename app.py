import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import predict_rating, predict_cluster, generate_flavor_profile, plot_pca_interactive, CLUSTER_NAMES, CLUSTER_DESCRIPTIONS
import pandas as pd
import plotly.express as px
import base64
import os

st.set_page_config(page_title="Coffee ML App", layout="centered")

st.markdown("""
<style>

:root {
    --coffee-primary: #6F4E37;     
    --coffee-secondary: #EDE0D4;     
    --coffee-light: #F7F3EF;        
    --text-dark: #3a3a3a;
    --text-light: #6e6e6e;
}

/* Global background image with opacity */
.stApp {
    background: 
        linear-gradient(rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.9)),
        url('https://cdn.pixabay.com/photo/2021/01/18/12/38/coffee-5928009_1280.jpg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
}

/* Ensure anchor sections are not hidden behind header */
[id] {
    scroll-margin-top: 60px !important;  /* adjust value as needed */
}

    /* ===== White content panel (your actual DOM container) ===== */
    .stMainBlockContainer {
        background: rgba(255,255,255,0.8);
        padding: 3rem 3rem 3rem 3rem;
        margin-top: 2rem;
    }
    .explore-section hr {
        display: none !important;
    }

/* Main content area */
.main .block-container {
    background-color: rgba(255, 255, 255, 0.98);
    border-radius: 10px;
    padding: 2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* input box background */
input[type="text"],
input[type="number"],
textarea,
div[data-baseweb="select"] > div {
    background-color: var(--coffee-secondary) !important;        
    border: 1px solid var(--coffee-primary) !important;       
    border-radius: 10px !important;
    color: var(--coffee-primary) !important;                  
}

[data-testid="stWidgetLabel"] > div:nth-child(1) {
    font-weight: 800 !important;  
    color: #6F4E37 !important;      
    font-size: 1.1rem !important;     
}    

/* Hover effect ONLY for cards */
.text-card:hover,
.roast-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    transition: all 0.25s ease;
}

     
/* title color - larger typography */
h1 {
    color: var(--coffee-primary) !important;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
}

h2 {
    color: var(--coffee-primary) !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

h3 {
    color: var(--coffee-primary) !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

h4 {
    color: var(--coffee-primary) !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
}

/* Section headers */
[data-testid="stHeader"] {
    color: var(--coffee-primary) !important;
}

/* default button - coffee-brown text */
.stButton button {
    background-color: var(--coffee-secondary) !important; 
    color: var(--coffee-primary) !important;
    font-weight: 700 !important;           
    border-radius: 8px !important;
    padding: 0.7em 1.2em !important;
    border: 2px solid var(--coffee-primary) !important;
    font-size: 1rem !important;
}

/* button hover  */
.stButton button:hover {
    background-color: var(--coffee-primary) !important;  
    color: white !important;
    font-weight: 800 !important;
    transform: scale(1.02);
    transition: all 0.2s ease;
}

div[data-testid="stNumberInput"] button {
    background-color: transparent !important;
    box-shadow: none !important;
    color: #3A3A3A !important;            
}

/* Section divider */
.section-divider {
    border-top: 2px solid var(--coffee-secondary);
    margin: 3rem 0;
    padding-top: 2rem;
}

/* Text-only card styling */
.text-card {
    background-color: white;
    border: 1px solid var(--coffee-secondary);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 4px rgba(111, 78, 55, 0.1);
}

.text-card h4 {
    color: var(--coffee-primary);
    margin-top: 0;
    font-weight: 700;
}

.text-card ul {
    color: var(--text-dark);
    line-height: 1.8;
    margin: 0;
    padding-left: 1.5rem;
}

.text-card li {
    margin-bottom: 0.5rem;
    font-size: 1rem;
}

.text-card p {
    color: var(--text-dark);
    line-height: 1.6;
    margin-bottom: 0.5rem;
}

/* Roast card with left text, right image */
.roast-card {
    background-color: white;
    border: 1px solid var(--coffee-secondary);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 4px rgba(111, 78, 55, 0.1);
    display: flex;
    gap: 1.5rem;
    align-items: center;
}

.roast-card-content {
    flex: 1;
}

.roast-card-img {
    width: 200px;
    height: 150px;
    object-fit: cover;
    border-radius: 8px;
    flex-shrink: 0;
}

.roast-card h4 {
    color: var(--coffee-primary);
    margin-top: 0;
    font-weight: 700;
}

.roast-card ul {
    color: var(--text-dark);
    line-height: 1.8;
    margin: 0;
    padding-left: 1.5rem;
}

.roast-card li {
    margin-bottom: 0.5rem;
    font-size: 1rem;
}

/* Tab styling - no filled background, underline for active */
[data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-radius: 0;
    padding: 0;
    border-bottom: 1px solid rgba(111, 78, 55, 0.2);
}

[data-baseweb="tab"] {
    color: var(--coffee-primary) !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    background-color: transparent !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.75rem 1.5rem !important;
    transition: border-bottom-color 0.2s ease !important;
}

[data-baseweb="tab"]:hover {
    border-bottom-color: rgba(111, 78, 55, 0.4) !important;
}

[data-baseweb="tab"][aria-selected="true"] {
    background-color: transparent !important;
    color: var(--coffee-primary) !important;
    border-bottom: 3px solid var(--coffee-primary) !important;
}

/* Background panel behind all main content */
.stApp > div:first-child::before {
    content: "";
    position: fixed;
    top: 120px;
    left: 50%;
    transform: translateX(-50%);
    width: 80%;
    max-width: 1400px;
    min-height: calc(100vh - 120px);
    background: #FAF7F3;
    border-radius: 20px;
    z-index: -1;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.08);
    pointer-events: none;
}

/* Ensure main content container has proper positioning */
.main .block-container {
    position: relative;
    z-index: 1;
}

div.section-divider {
    display: none !important;
}
/* ===== Left Sidebar Navigation ===== */
.explore-sidebar {
    position: fixed;
    top: 180px;                         /* 你可以调 */
    left: 40px;                         /* 你可以调 */
    width: 200px;                       /* sidebar 宽度 */
    background: rgba(255,255,255,0.65); 
    backdrop-filter: blur(4px);
    border: 1px solid #E6DCD3;
    border-radius: 12px;
    padding: 1rem 1rem;
    box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    z-index: 999;
}

.explore-sidebar a {
    display: block;
    font-size: 1rem;
    font-weight: 700;
    color: #6F4E37;
    margin-bottom: 1rem;
    text-decoration: none;
    transition: all 0.2s ease;
}

.explore-sidebar a:hover {
    color: #FFFFFF;
    background: #6F4E37;
    padding-left: 8px;
    border-radius: 8px;
}


}

</style>
""", unsafe_allow_html=True)

st.title("☕ Coffee Flavor & Rating Predictor")

# Reusable card functions
def text_card(title, bullets):
    """Create a text-only coffee card"""
    bullet_html = "".join([f"<li>{b}</li>" for b in bullets])
    st.markdown(
        f"""
        <div class="text-card">
            <h4>{title}</h4>
            <ul>{bullet_html}</ul>
        </div>
        """,
        unsafe_allow_html=True
    )

def roast_card(title, bullets, image_path):
    """Create a roast card with left text and right image"""
    bullet_html = "".join([f"<li>{b}</li>" for b in bullets])
    
    # Encode image to base64 for HTML display
    img_base64 = ""
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
            img_base64 = f"data:image/jpeg;base64,{img_base64}"
    else:
        # Fallback if image doesn't exist
        img_base64 = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgZmlsbD0iI0Y3RjNFRiIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM2RjRFMzciIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5JbWFnZSBub3QgZm91bmQ8L3RleHQ+PC9zdmc+"
    
    st.markdown(
        f"""
        <div class="roast-card">
            <div class="roast-card-content">
                <h4>{title}</h4>
                <ul>{bullet_html}</ul>
            </div>
            <img src="{img_base64}" class="roast-card-img" alt="{title}">
        </div>
        """,
        unsafe_allow_html=True
    )

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# Data load
df = pd.read_csv("data/df_for_pca.csv")

# Load PCA data for map
pca_data = np.load("models/pca_data.npy")
cluster_labels = np.load("models/cluster_labels.npy")
cluster_labels = np.array(cluster_labels, dtype=int)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Explore", "Predict", "History"])

# =======================
#   TAB 1: EXPLORE
# =======================
with tab1:
    # --- Sidebar Navigation HTML ---
    st.markdown("""
    <div class="explore-sidebar">
        <a href="#bean-species">Coffee Bean Species</a>
        <a href="#roast-levels">Roast Levels</a>
        <a href="#brew-methods">Brew Methods</a>
        <a href="#ml-clusters">ML Flavor Clusters</a>
        <a href="#global-distribution">Global Distribution</a>
    </div>
    """, unsafe_allow_html=True)

    st.header("🌍 Explore Coffee Knowledge")
    
    # =======================
    #   SECTION 1: Coffee Bean Species
    # =======================
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div id="bean-species"></div>', unsafe_allow_html=True)
    st.subheader("📔 Coffee Bean Species")
    
    text_card("Arabica", [
        "More complex and refined flavors",
        "Notes of fruit, flowers, and sweetness",
        "Brighter acidity",
        "Lower caffeine",
        "Higher quality, widely used in specialty coffee",
        "Best for: people who enjoy nuanced, aromatic coffee."
    ])
    
    text_card("Robusta", [
        "Stronger, more bitter taste",
        "Earthy, smoky, heavy body",
        "Much higher caffeine",
        "Cheaper and more disease-resistant",
        "Often used in espresso blends and instant coffee",
        "Best for: those who prefer bold, intense coffee."
    ])
    
    # =======================
    #   SECTION 2: Roast Levels
    # =======================
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2 id="roast-levels"></div>', unsafe_allow_html=True)
    st.subheader("🔥 Roast Levels")
    
    roast_card("Light Roast", [
        "Bright, fruity, floral",
        "Higher acidity",
        "Highlights bean origin flavors"
    ], "./Pictures/light.jpg")
    
    roast_card("Medium Roast", [
        "Balanced sweetness and acidity",
        "Rounder body",
        "Notes of nuts, caramel, chocolate"
    ], "./Pictures/medium.jpg")
    
    roast_card("Dark Roast", [
        "Bold, smoky, bitter",
        "Low acidity",
        "More \"roast flavor,\" less origin character"
    ], "./Pictures/dark.jpg")
    
    # =======================
    #   SECTION 3: Brew Methods
    # =======================
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div id="brew-methods"></div>', unsafe_allow_html=True)
    st.subheader("🟤 Brew Methods")
    
    text_card("French Press", [
        "Full-bodied, rich coffee with natural oils",
        "Coarse grind, 4-minute steep time"
    ])
    
    text_card("Pour Over", [
        "Clean, bright, and nuanced flavors",
        "Precise water control highlights origin characteristics"
    ])
    
    text_card("Cold Brew", [
        "Smooth, low-acidity coffee",
        "Coarse grind steeped in cold water for 12-24 hours"
    ])
    
    text_card("Espresso", [
        "Concentrated, intense coffee with crema",
        "High pressure extraction in 25-30 seconds"
    ])

    text_card("Moka Pot", [
        "Strong, intense, espresso-like flavor",
        "High pressure for home brewing in about 2-5 minutes"
    ])
    
    # =======================
    #   SECTION 4: ML Flavor Clusters Overview
    # =======================
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div id="ml-clusters"></div>', unsafe_allow_html=True)
    st.subheader("🗂️ ML Flavor Clusters Overview")
    
    # Initialize selected cluster for map
    if "explore_selected_cluster" not in st.session_state:
        st.session_state["explore_selected_cluster"] = 0
    
    # Display cluster cards vertically
    for cluster_id, cluster_name in CLUSTER_NAMES.items():
        cluster_desc = CLUSTER_DESCRIPTIONS.get(cluster_id, "")
        text_card(f"{cluster_name}", [cluster_desc])
    
    # =======================
    #   SECTION 5: Cluster Global Distribution Map
    # =======================
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div id="global-distribution"></div>', unsafe_allow_html=True)
    st.subheader("🌍 Global Distribution of Coffee Clusters")
    
    cluster_names = CLUSTER_NAMES
    
    # Cluster selection buttons
    c1, c2, c3 = st.columns(3)
    if c1.button(cluster_names[0], key="explore_btn_0"):
        st.session_state["explore_selected_cluster"] = 0
    if c2.button(cluster_names[1], key="explore_btn_1"):
        st.session_state["explore_selected_cluster"] = 1
    if c3.button(cluster_names[2], key="explore_btn_2"):
        st.session_state["explore_selected_cluster"] = 2
    
    c4, c5, c6 = st.columns(3)
    if c4.button(cluster_names[3], key="explore_btn_3"):
        st.session_state["explore_selected_cluster"] = 3
    if c5.button(cluster_names[4], key="explore_btn_4"):
        st.session_state["explore_selected_cluster"] = 4
    if c6.button(cluster_names[5], key="explore_btn_5"):
        st.session_state["explore_selected_cluster"] = 5
    
    selected_cluster = st.session_state["explore_selected_cluster"]
    
    # Filter df by cluster
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
            title=f"{cluster_names[selected_cluster]} — Global Distribution",
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

# =======================
#   TAB 2: PREDICT
# =======================
with tab2:
    st.header("🔮 Predict Coffee Rating & Flavor")
    st.write("Input the coffee information, I will predict the rating and determine which flavor cluster it belongs to.")
    
    # =======================
    #   User Inputs
    # =======================
    
    st.subheader("📥 Input the coffee information")
    
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
    loc_list = ['Hong Kong', 'United States', 'Canada', 'Taiwan', "Hawai'i", 'Australia',
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
        
        # --- Save to History ---
        coffee_name = st.session_state.get("coffee_name", "Unnamed Coffee")
        st.session_state.history.append({
            "name": coffee_name,
            "roast": roast,
            "loc": loc,
            "price": price,
            "rating": predicted_rating,
            "cluster": cluster_name
        })
    
    
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
        
        # ---------------------------
        # 3. GENERATE AI ANALYSIS BUTTON
        # ---------------------------
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

# =======================
#   TAB 3: HISTORY
# =======================
with tab3:
    st.header("📜 Prediction History")
    
    if len(st.session_state.history) == 0:
        st.info("No predictions yet. Make a prediction in the Predict tab to see your history here.")
    else:
        # Display as dataframe
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        
        # Also display as cards for a more polished look
        st.subheader("📋 Detailed History")
        for idx, entry in enumerate(reversed(st.session_state.history)):
            st.markdown(f"""
            <div class="coffee-card">
                <h3>{entry['name']}</h3>
                <p><strong>Rating:</strong> {entry['rating']:.2f} / 100</p>
                <p><strong>Cluster:</strong> {entry['cluster']}</p>
                <p><strong>Roast:</strong> {entry['roast']} | <strong>Location:</strong> {entry['loc']} | <strong>Price:</strong> ${entry['price']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
