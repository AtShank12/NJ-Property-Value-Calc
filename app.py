import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("final_combined_data.csv")
    df['town'] = df['town'].astype(str).str.strip().str.lower()
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['price', 'year'])
    return df

df = load_data()

# Train Linear + Polynomial models for each town
town_models = {}
for town in df['town'].unique():
    subset = df[df['town'] == town]
    if len(subset) > 2:
        X = subset[['year']]
        y = subset['price']

        linear = LinearRegression()
        poly = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

        linear.fit(X, y)
        poly.fit(X, y)

        town_models[town] = {'linear': linear, 'poly': poly}

# Erosion Index (complete)
erosion_index = {
    'allenhurst': 6.5, 'asbury park': 6.8, 'atlantic city': 8.2, 'avalon': 7.9, 'barnegat light': 6.2,
    'bay head': 7.4, 'beach haven': 7.8, 'belmar': 6.7, 'brigantine city': 8.0, 'cape may city': 6.0,
    'deal': 7.2, 'harvey cedars': 6.3, 'highlands': 9.0, 'holgate': 7.5, 'interlaken': 5.5,
    'island heights': 6.1, 'keansburg': 7.1, 'keyport': 6.5, 'lavallette': 7.6, 'long beach': 7.0,
    'long branch': 7.4, 'longport': 6.8, 'manasquan': 7.3, 'mantoloking': 8.3, 'margate': 6.4,
    'monmouth beach': 7.6, 'neptune city': 5.5, 'ocean city': 8.1, 'ocean gate': 5.9, 'pine beach': 6.0,
    'point pleasant beach': 6.9, 'sea bright': 9.1, 'sea girt': 7.2, 'sea isle city': 8.0,
    'seaside heights': 8.5, 'seaside park': 7.5, 'spring lake': 7.3, 'spring lake heights': 6.6,
    'surf city': 6.4, 'union beach': 8.0, 'ventnor': 6.6, 'wildwood': 9.2, 'wildwood crest': 8.7
}
default_score = 6.5
for town in df['town'].unique():
    if town not in erosion_index:
        erosion_index[town] = default_score

# Streamlit interface
st.title("Shank's Calc")

user_input = st.text_input("Enter your town/city with the year you want to get your property value in. It's like magic!!!")

if user_input:
    match = re.search(r'(\d{4})', user_input)
    year = int(match.group(1)) if match else None

    matched_town = None
    for town in town_models:
        if town in user_input.lower():
            matched_town = town
            break

    if matched_town and year:
        models = town_models[matched_town]
        X_input = np.array([[year]])

        linear_pred = models['linear'].predict(X_input)[0]
        poly_pred = models['poly'].predict(X_input)[0]
        prediction = np.mean([linear_pred, poly_pred])

        st.success(f"🏘️ Predicted average property value in **{matched_town.title()}** for **{year}**: **${prediction:,.2f}**")

        # Erosion Index
        erosion = erosion_index[matched_town]
        if erosion >= 8:
            level = "🌊 Highly Prone to Erosion"
        elif erosion >= 6:
            level = "⚠️ Moderately Prone to Erosion"
        else:
            level = "✅ Low Erosion Risk"

        st.metric("Erosion Index", f"{erosion}/10", help="1 = Low risk, 10 = High risk")
        st.info(f"This town is classified as: **{level}**")

        # Clean and styled property trend plot
        town_data = df[df['town'] == matched_town].sort_values(by='year')
        if not town_data.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(town_data['year'], town_data['price'], marker='o', linestyle='-', linewidth=2)

            ax.set_title(f"📈 Property Value Trend in {matched_town.title()}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Year", fontsize=12)
            ax.set_ylabel("Average Sold Price ($)", fontsize=12)

            min_year = 2000
            max_year = max(town_data['year'].max(), year)
            ax.set_xlim(min_year, max_year + 1)

            year_range = max_year - min_year
            tick_step = 5 if year_range > 20 else 3
            ax.set_xticks(np.arange(min_year, max_year + 1, step=tick_step))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            ax.grid(True, linestyle='--', alpha=0.5)
            ax.tick_params(axis='both', labelsize=10)
            fig.tight_layout()

            st.pyplot(fig)
    else:
        st.error("❌ Please enter a valid town and year.")

# Supported towns list
with st.expander("📍 View all supported towns"):
    st.write(", ".join(sorted(town_models.keys())))
